import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
import sys
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import warnings
import time
import signal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError, as_completed
import json
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from functools import lru_cache
import multiprocessing as mp
from threading import Lock
from typing import List

# 多进程支持：确保在不同环境下正常工作
def setup_multiprocessing():
    """设置多进程启动方法"""
    try:
        # 尝试设置spawn方法（更安全，但可能较慢）
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
            print(f"设置多进程启动方法为: spawn")
        else:
            print(f"当前多进程启动方法: {mp.get_start_method()}")
    except RuntimeError as e:
        print(f"多进程设置警告: {e}")
    except Exception as e:
        print(f"多进程设置错误: {e}")

# 过滤警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 尝试导入泰语分词工具
try:
    from pythainlp import word_tokenize as thai_tokenize
    THAI_TOKENIZER_AVAILABLE = True
except ImportError:
    THAI_TOKENIZER_AVAILABLE = False
    print("警告：未安装 pythainlp，泰语将使用字符级分词")

# 全局常量，确保子进程能访问
THAI_AVAILABLE = THAI_TOKENIZER_AVAILABLE

dev_mode = True

# 路径配置
base_dir = os.getcwd()
shot_path = os.path.join(base_dir, "data/text_data/train.csv")
if dev_mode:
    input_path = os.path.join(base_dir, "data/text_data/dev.csv")
else:
    input_path = os.path.join(base_dir, "output/testa_asr.csv")

# BLEU2缓存配置
cache_dir = os.path.join(base_dir, f"output/cache{'_dev' if 'dev' in input_path else ''}")
bleu2_cache_file = os.path.join(cache_dir, "bleu2_cache.json")
# 分词缓存文件
tokenize_cache_file = os.path.join(cache_dir, "tokenize_cache.json")
# 相似度排序缓存文件
similarity_ranking_cache_file = os.path.join(cache_dir, "similarity_ranking_cache.json")

def get_output_path(n_shot):
    if dev_mode:
        return os.path.join(base_dir, f"output/gemini/dev_gp_{n_shot}shot.csv" if n_shot > 0 else "output/gemini/dev_g.csv")
    else:
        return os.path.join(base_dir, f"output/gemini/result.csv")

# 语言映射
lang_map = {
    "马来语": "Malay",
    "泰语": "Thai", 
    "英语": "English"
}

# Gemini API配置
API_KEYS = ["AIzaSyAfOE7TG4C6mYF-en8DVUMKrU-GgAOaAX0", "AIzaSyDzEM-8qtZLYSjxK9OxUiXvjm8v96nKX3g", "AIzaSyAUyX_M64ZsLM7ds2hTEMCTF_CE4ySLPZg", "AIzaSyCIg0DuyUkrfPWGbhwJ5Jq-SMMAj9WbO20", "AIzaSyBHbYVAi1ewtp88teVnu_zLXoJU1ORk3gE"]
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.5-pro-preview-06-05"

# 并发配置
MAX_CONCURRENT_REQUESTS = len(API_KEYS)  # 线程数等于API key数量
REQUEST_DELAY = 1  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_FACTOR = 2  # 退避因子
# Few-shot 配置
DEFAULT_N_SHOT = 3  # 默认示例数量

# 性能优化配置
BLEU2_WORKERS = min(mp.cpu_count(), 96)  # BLEU2计算的并行工作进程数
BATCH_SIZE = 1000  # 批量处理大小

# 全局分词缓存
_tokenize_cache = {}

# Gemini API key 轮询管理
class GeminiKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self._current_index = 0
        self._lock = Lock()
        print(f"初始化Gemini Key Manager，共 {len(api_keys)} 个API keys")
    
    def get_next_api_key(self) -> str:
        """使用轮询方式获取下一个API密钥"""
        with self._lock:
            api_key = self.api_keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.api_keys)
            return api_key

# 全局key管理器
key_manager = GeminiKeyManager(API_KEYS)

def save_similarity_ranking_cache(similarity_ranking_cache):
    """保存相似度排序缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # 转换索引为字符串以便JSON序列化
        cache_for_json = {}
        for dev_idx, lang_rankings in similarity_ranking_cache.items():
            cache_for_json[str(dev_idx)] = {}
            for lang, train_idx_list in lang_rankings.items():
                cache_for_json[str(dev_idx)][lang] = [str(idx) for idx in train_idx_list]
        
        with open(similarity_ranking_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_for_json, f, ensure_ascii=False, indent=2)
        print(f"已保存相似度排序缓存，包含 {len(similarity_ranking_cache)} 个开发样本的排序")
    except Exception as e:
        print(f"保存相似度排序缓存失败: {e}")

def load_similarity_ranking_cache():
    """加载相似度排序缓存"""
    if not os.path.exists(similarity_ranking_cache_file):
        return None
    
    try:
        with open(similarity_ranking_cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 转换字符串索引回整数
        similarity_ranking_cache = {}
        for dev_idx_str, lang_rankings in cache_data.items():
            dev_idx = int(dev_idx_str)
            similarity_ranking_cache[dev_idx] = {}
            for lang, train_idx_list in lang_rankings.items():
                similarity_ranking_cache[dev_idx][lang] = [int(idx) for idx in train_idx_list]
        
        print(f"成功加载相似度排序缓存，包含 {len(similarity_ranking_cache)} 个开发样本的排序")
        return similarity_ranking_cache
    except Exception as e:
        print(f"加载相似度排序缓存失败: {e}")
        return None

def tokenize_for_bleu_cached(text, language):
    """带缓存的分词函数"""
    # 创建缓存键
    cache_key = f"{language}:{text}"
    
    # 检查缓存
    if cache_key in _tokenize_cache:
        return _tokenize_cache[cache_key]
    
    # 执行分词
    if language in ["Thai", "泰语"]:
        if THAI_TOKENIZER_AVAILABLE:
            # 使用泰语分词工具
            try:
                tokens = thai_tokenize(text, engine='newmm')
            except Exception as e:
                # 如果分词失败，使用字符级分词（去除空格）
                tokens = list(text.replace(" ", ""))
        else:
            # 如果没有安装pythainlp，使用字符级分词
            tokens = list(text.replace(" ", ""))
    elif language in ["English", "英语"]:
        # 英语使用空格分词
        tokens = text.split()
    elif language in ["Malay", "马来语"]:
        # 马来语也使用空格分词
        tokens = text.split()
    elif language in ["中文", "Chinese"]:
        # 中文使用字符级分词
        tokens = list(text.replace(" ", ""))
    else:
        # 默认使用空格分词
        tokens = text.split()
    
    # 缓存结果
    _tokenize_cache[cache_key] = tokens
    return tokens

def calculate_bleu2_similarity_fast(tokens1, tokens2):
    """快速计算两个已分词文本的BLEU2相似度分数"""
    try:
        if len(tokens1) > 0 and len(tokens2) > 0:
            # 使用BLEU2 (bigram)，权重为(0.5, 0.5, 0, 0)，加入平滑函数减少警告
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([tokens1], tokens2, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            return bleu_score
        else:
            return 0.0
    except Exception as e:
        return 0.0

def calculate_bleu2_similarity(text1, text2, language="中文"):
    """计算两个文本之间的BLEU2相似度分数"""
    try:
        tokens1 = tokenize_for_bleu_cached(text1, language)
        tokens2 = tokenize_for_bleu_cached(text2, language)
        return calculate_bleu2_similarity_fast(tokens1, tokens2)
    except Exception as e:
        print(f"BLEU2计算失败: {e}")
        return 0.0

def save_bleu2_cache(bleu2_cache):
    """保存BLEU2缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # 转换索引为字符串以便JSON序列化
        cache_for_json = {}
        for dev_idx, train_scores in bleu2_cache.items():
            cache_for_json[str(dev_idx)] = {str(train_idx): score for train_idx, score in train_scores.items()}
        
        with open(bleu2_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_for_json, f, ensure_ascii=False, indent=2)
        print(f"已保存BLEU2缓存，包含 {len(bleu2_cache)} 个开发样本的分数")
    except Exception as e:
        print(f"保存BLEU2缓存失败: {e}")

def load_bleu2_cache():
    """加载BLEU2缓存"""
    if not os.path.exists(bleu2_cache_file):
        print(f"BLEU2缓存文件不存在: {bleu2_cache_file}")
        print("请先运行预计算: python infer_by_gemini.py --precompute-bleu2")
        return None
    
    try:
        with open(bleu2_cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 转换字符串索引回整数
        bleu2_cache = {}
        for dev_idx_str, train_scores in cache_data.items():
            dev_idx = int(dev_idx_str)
            bleu2_cache[dev_idx] = {int(train_idx_str): score for train_idx_str, score in train_scores.items()}
        
        print(f"成功加载BLEU2缓存，包含 {len(bleu2_cache)} 个开发样本的分数")
        return bleu2_cache
    except Exception as e:
        print(f"加载BLEU2缓存失败: {e}")
        return None

class GeminiTranslator:
    def __init__(self, api_keys, base_url, model_name, n_shot=0):
        self.api_keys = api_keys
        self.base_url = base_url
        self.model_name = model_name
        self.n_shot = n_shot
        self.examples_cache = {}  # 缓存示例数据
        self.bleu2_cache = None  # BLEU2分数缓存
        self.similarity_ranking_cache = None  # 相似度排序缓存
        self.train_data = None  # 训练数据缓存
        
        # 加载缓存和训练数据
        if self.n_shot > 0:
            self.bleu2_cache = load_bleu2_cache()
            self.similarity_ranking_cache = load_similarity_ranking_cache()
            # 预加载训练数据到内存
            if os.path.exists(shot_path):
                print("加载训练数据到内存...")
                self.train_data = pd.read_csv(shot_path)
                print(f"训练数据加载完成：{len(self.train_data)} 条")
            else:
                print(f"警告：训练数据文件不存在: {shot_path}")
        
    def load_examples(self, target_lang, current_text=None, dev_idx=None):
        """从训练数据加载指定语言的翻译示例，基于缓存的相似度排序选择最相似的示例"""
        # 为缓存添加dev_idx来区分不同的查询
        cache_key = f"{target_lang}_{dev_idx if dev_idx is not None else hash(current_text) if current_text else 'default'}"
        
        if cache_key in self.examples_cache:
            return self.examples_cache[cache_key]
            
        examples = []
        if self.n_shot > 0 and self.train_data is not None:
            try:
                # 筛选出目标语言的数据，且有有效翻译结果的
                lang_data = self.train_data[
                    (self.train_data['语言'] == target_lang) &  # 语言列匹配
                    (self.train_data['文本'].notna()) &  # 有翻译结果
                    (self.train_data['文本'] != "") &  # 翻译结果非空
                    (self.train_data['中文'].notna()) &  # 中文文本非空
                    (self.train_data['中文'] != "")
                ]
                
                if len(lang_data) >= self.n_shot:
                    if dev_idx is not None and self.similarity_ranking_cache and dev_idx in self.similarity_ranking_cache:
                        # 优先使用相似度排序缓存（最快）
                        if target_lang in self.similarity_ranking_cache[dev_idx]:
                            sorted_train_indices = self.similarity_ranking_cache[dev_idx][target_lang]
                            # 筛选出在当前lang_data中存在的索引，并选择top n_shot个
                            lang_data_indices = set(lang_data.index)  # 转换为set提高查找速度
                            valid_indices = [idx for idx in sorted_train_indices if idx in lang_data_indices]
                            selected_indices = valid_indices[:self.n_shot]
                            
                            for train_idx in selected_indices:
                                row = self.train_data.loc[train_idx]
                                chinese_text = row['中文']  # 中文列
                                translated_text = row['文本']  # 翻译结果
                                examples.append((chinese_text, translated_text))
                    elif dev_idx is not None and self.bleu2_cache and dev_idx in self.bleu2_cache:
                        # 备选方案：使用BLEU2缓存并重新排序（较慢）
                        cached_scores = self.bleu2_cache[dev_idx]
                        
                        # 获取该语言的训练数据分数
                        lang_scores = []
                        for train_idx in lang_data.index:
                            if train_idx in cached_scores:
                                bleu_score = cached_scores[train_idx]
                                lang_scores.append((bleu_score, train_idx))
                        
                        # 按BLEU2分数降序排序，选择top n_shot个
                        lang_scores.sort(key=lambda x: x[0], reverse=True)
                        selected_indices = [train_idx for _, train_idx in lang_scores[:self.n_shot]]
                        
                        for train_idx in selected_indices:
                            row = self.train_data.loc[train_idx]
                            chinese_text = row['中文']  # 中文列
                            translated_text = row['文本']  # 翻译结果
                            examples.append((chinese_text, translated_text))
                            
                    elif current_text:
                        # 实时计算BLEU2分数（备用方案）
                        print(f"✗ 缓存未命中，实时计算BLEU2分数（慢）")
                        bleu_scores = []
                        for _, row in lang_data.iterrows():
                            chinese_text = row['中文']
                            bleu_score = calculate_bleu2_similarity(current_text, chinese_text, "中文")
                            bleu_scores.append((bleu_score, row))
                        
                        # 按BLEU2分数降序排序，选择top n_shot个
                        bleu_scores.sort(key=lambda x: x[0], reverse=True)
                        selected_rows = [item[1] for item in bleu_scores[:self.n_shot]]
                        
                        print(f"实时计算BLEU2分数: {[round(item[0], 4) for item in bleu_scores[:self.n_shot]]}")
                        
                        for row in selected_rows:
                            chinese_text = row['中文']  # 中文列
                            translated_text = row['文本']  # 翻译结果
                            examples.append((chinese_text, translated_text))
                    else:
                        # 如果没有current_text，随机选择（兼容旧逻辑）
                        sampled_data = lang_data.sample(n=self.n_shot, random_state=42)
                        for _, row in sampled_data.iterrows():
                            chinese_text = row['中文']  # 中文列
                            translated_text = row['文本']  # 翻译结果
                            examples.append((chinese_text, translated_text))
                else:
                    # 如果数据不够，使用所有可用的
                    if dev_idx is not None and self.similarity_ranking_cache and dev_idx in self.similarity_ranking_cache:
                        # 优先使用相似度排序缓存
                        if target_lang in self.similarity_ranking_cache[dev_idx]:
                            sorted_train_indices = self.similarity_ranking_cache[dev_idx][target_lang]
                            # 筛选出在当前lang_data中存在的索引，使用所有可用的
                            lang_data_indices = set(lang_data.index)  # 转换为set提高查找速度
                            valid_indices = [idx for idx in sorted_train_indices if idx in lang_data_indices]
                            
                            for train_idx in valid_indices:
                                row = self.train_data.loc[train_idx]
                                chinese_text = row['中文']
                                translated_text = row['文本']
                                examples.append((chinese_text, translated_text))
                    else:
                        # 使用所有可用的数据
                        for _, row in lang_data.iterrows():
                            chinese_text = row['中文']
                            translated_text = row['文本']
                            examples.append((chinese_text, translated_text))
                        
            except Exception as e:
                print(f"加载示例时出错: {e}")
                examples = []
        
        self.examples_cache[cache_key] = examples
        return examples
        
    def create_system_prompt(self, source_lang, target_lang, current_text=None, dev_idx=None):
        """创建系统提示词，包含 few-shot 示例"""
        if self.n_shot == 0:
            # 原有的系统提示词
            system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。用户将输入中文文本，请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
        else:
            # 包含示例的系统提示词，基于缓存的BLEU2分数选择
            examples = self.load_examples(target_lang, current_text, dev_idx)
            
            if examples:
                examples_text = "\n\n以下是一些翻译示例供参考（按相似度排序）：\n"
                for i, (chinese, translated) in enumerate(examples, 1):
                    examples_text += f"示例{i}:\n"
                    examples_text += f"中文: {chinese}\n"
                    examples_text += f"{target_lang}: {translated}\n\n"
                
                system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。{examples_text}请参考以上示例的翻译风格和质量，将用户输入的中文文本翻译成{target_lang}。请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
            else:
                # 如果没有找到示例，使用原有提示词
                system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。用户将输入中文文本，请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
                
        return system_prompt
    
    def create_translation_prompt(self, text):
        """创建翻译提示词"""
        return text
    
    def translate_text(self, text, source_lang, target_lang, dev_idx=None):
        """使用Gemini API翻译文本，带重试和指数退避"""
        # 预先检查输入数据的有效性
        if not text or pd.isna(text) or text.strip() == "":
            print(f"输入文本为空或无效，跳过翻译")
            return ""
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 获取API key并创建客户端
                api_key = key_manager.get_next_api_key()
                client = OpenAI(
                    api_key=api_key,
                    base_url=self.base_url
                )
                
                # 创建系统和用户提示，传入dev_idx用于示例选择
                system_prompt = self.create_system_prompt(source_lang, target_lang, text, dev_idx)
                user_prompt = self.create_translation_prompt(text)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    reasoning_effort="low",  # 不思考
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4096,  # 设置为最大新生成token数
                    stream=False,
                    timeout=30  # 添加30秒超时
                )
                
                # 安全地获取翻译结果，处理None情况
                if (response and response.choices and len(response.choices) > 0 and 
                    response.choices[0].message and response.choices[0].message.content):
                    translated_text = response.choices[0].message.content.strip()
                else:
                    print(f"API响应格式异常或内容为空")
                    # 添加详细的调试信息
                    print(f"  调试信息:")
                    print(f"    - response存在: {response is not None}")
                    if response:
                        print(f"    - response.choices存在: {hasattr(response, 'choices') and response.choices is not None}")
                        if hasattr(response, 'choices') and response.choices:
                            print(f"    - choices长度: {len(response.choices)}")
                            if len(response.choices) > 0:
                                print(f"    - first choice存在: {response.choices[0] is not None}")
                                if response.choices[0]:
                                    print(f"    - message存在: {hasattr(response.choices[0], 'message') and response.choices[0].message is not None}")
                                    if hasattr(response.choices[0], 'message') and response.choices[0].message:
                                        print(f"    - content存在: {hasattr(response.choices[0].message, 'content')}")
                                        print(f"    - content值: {response.choices[0].message.content}")
                    translated_text = ""
                
                time.sleep(REQUEST_DELAY)  # 添加请求延迟
                return translated_text
                
            except Exception as e:
                error_msg = str(e)
                print(f"翻译过程中出错: {error_msg}")
                
                # 添加更详细的错误信息用于调试
                if "NoneType" in error_msg:
                    print(f"  -> 这可能是API响应为空导致的，正在重试...")
                elif "timeout" in error_msg.lower():
                    print(f"  -> 请求超时，可能是网络连接问题")
                elif "rate limit" in error_msg.lower():
                    print(f"  -> API速率限制，正在等待更长时间...")
                    time.sleep(10)  # 速率限制时等待更长时间
                
                if attempt < MAX_RETRIES:
                    wait_time = REQUEST_DELAY * (BACKOFF_FACTOR ** attempt)
                    print(f"等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"达到最大重试次数，跳过当前文本")
                    return ""
        
        return ""

def translate_batch_parallel(translator, texts_and_langs_with_idx, max_workers, pbar=None):
    """使用线程池并行翻译文本，带超时机制防止卡住"""
    results = [""] * len(texts_and_langs_with_idx)
    
    def translate_single(index_and_data):
        index, (text, source_lang, target_lang, dev_idx) = index_and_data
        if pd.isna(text) or text == "" or text is None:
            return index, ""
        
        # 添加单个任务的调试信息
        try:
            result = translator.translate_text(text, source_lang, target_lang, dev_idx)
            # 确保结果不为None
            if result is None:
                result = ""
            return index, result
        except Exception as e:
            print(f"任务 {index} (dev_idx={dev_idx}) 翻译失败: {e}")
            return index, ""
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        indexed_data = [(i, data) for i, data in enumerate(texts_and_langs_with_idx)]
        
        # 提交所有任务
        future_to_index = {}
        for item in indexed_data:
            future = executor.submit(translate_single, item)
            future_to_index[future] = item[0]  # 保存任务索引
        
        # 使用as_completed来处理完成的任务，避免等待特定顺序
        completed_count = 0
        failed_count = 0
        
        for future in as_completed(future_to_index, timeout=300):  # 整个批次最多等待5分钟
            task_index = future_to_index[future]
            try:
                # 对每个任务设置较短的超时
                index, result = future.result(timeout=60)  # 单个任务最多等待1分钟
                results[index] = result
                completed_count += 1
                
                if pbar:
                    pbar.update(1)
                    
            except TimeoutError:
                print(f"任务 {task_index} 超时（>60秒），跳过")
                results[task_index] = ""  # 超时的任务返回空字符串
                failed_count += 1
                if pbar:
                    pbar.update(1)
            except Exception as e:
                print(f"任务 {task_index} 出错: {e}")
                results[task_index] = ""
                failed_count += 1
                if pbar:
                    pbar.update(1)
        
    return results

def signal_handler(signum, frame):
    """信号处理器，用于调试卡住的问题"""
    print(f"\n收到信号 {signum}，正在强制终止...")
    print("如果经常在同一位置卡住，请检查网络连接或降低并发数")
    sys.exit(1)

def translate_gemini(n_shot=0):
    """主翻译函数"""
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    print("开始使用Gemini API进行翻译...")
    print("提示：如果程序卡住，可以按 Ctrl+C 强制终止")
    if n_shot > 0:
        print(f"使用 {n_shot}-shot 学习模式")
    
    # 获取输出路径
    gemini_output_path = get_output_path(n_shot)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(gemini_output_path), exist_ok=True)
    
    # 检查API密钥
    if API_KEYS[0] == "your-api-key-1":
        print("警告：请在脚本中设置您的Gemini API密钥！")
        print("请将API_KEYS列表替换为您的实际API密钥")
        return
    
    # 读取ASR结果文件
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        print("请先运行ASR处理生成输入文件")
        return
    
    df = pd.read_csv(input_path)
    print(f"读取到 {len(df)} 条数据")
    
    # 检查结果文件是否存在，如果存在则加载已有结果
    if os.path.exists(gemini_output_path):
        print(f"发现已存在结果文件: {gemini_output_path}")
        existing_df = pd.read_csv(gemini_output_path)
        print(f"已存在结果文件包含 {len(existing_df)} 条数据")
        
        # 确保两个文件的行数匹配
        if len(existing_df) == len(df):
            # 如果行数匹配，直接使用已存在文件的answer列
            if 'answer' in existing_df.columns:
                df['answer'] = existing_df['answer']
                completed_count = len(existing_df[existing_df['answer'].notna() & (existing_df['answer'] != "")])
                print(f"从已存在文件中加载了 {completed_count} 条已完成的翻译结果")
            else:
                print("已存在文件中没有answer列，将从头开始翻译")
                df['answer'] = ""
        else:
            print(f"警告：已存在文件行数({len(existing_df)})与输入文件行数({len(df)})不匹配")
            print("将从头开始翻译")
            df['answer'] = ""
    else:
        print("未发现已存在的结果文件，将从头开始翻译")
        df['answer'] = ""
    
    # 获取需要翻译的数据
    need_translation = df[df['answer'].isna() | (df['answer'] == "")]
    
    if len(need_translation) == 0:
        print("所有数据已翻译完成！")
        return
        
    print(f"需要翻译 {len(need_translation)} 条数据")
    
    # 准备翻译数据（包含dev_idx）
    texts_and_langs_with_idx = []
    indices = []
    
    for idx, row in need_translation.iterrows():
        chinese_text = row.iloc[4]  # 中文列
        target_lang_chinese = row.iloc[3]  # 语言列
        target_lang_english = lang_map.get(target_lang_chinese, target_lang_chinese)
        
        texts_and_langs_with_idx.append((chinese_text, "中文", target_lang_chinese, idx))  # 添加dev_idx
        indices.append(idx)
    
    # 创建Gemini翻译器
    translator = GeminiTranslator(API_KEYS, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    # 分批翻译，每批保存一次
    batch_size = MAX_CONCURRENT_REQUESTS
    total_batches = (len(texts_and_langs_with_idx) + batch_size - 1) // batch_size
    
    print(f"开始翻译，共 {total_batches} 批，每批 {batch_size} 条...")
    print(f"使用 {MAX_CONCURRENT_REQUESTS} 个并发线程（等于API key数量）...")
    
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    
    # 创建总进度条
    with tqdm(total=len(texts_and_langs_with_idx), desc="总翻译进度") as pbar:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(texts_and_langs_with_idx))
            batch_texts_and_langs = texts_and_langs_with_idx[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]
            
            # 翻译当前批次
            batch_start_time = time.time()
            try:
                batch_results = translate_batch_parallel(translator, batch_texts_and_langs, MAX_CONCURRENT_REQUESTS, pbar)
            except Exception as e:
                print(f"批次 {batch_idx + 1} 处理出错: {e}")
                # 创建空结果继续处理
                batch_results = [""] * len(batch_texts_and_langs)
            batch_end_time = time.time()
            
            # 更新当前批次的结果到DataFrame
            batch_successful = 0
            batch_failed = 0
            
            for i, result in enumerate(batch_results):
                if result:  # 只更新非空结果
                    df.loc[batch_indices[i], 'answer'] = result
                    batch_successful += 1
                else:
                    batch_failed += 1
            
            # 保存当前进度
            df.to_csv(gemini_output_path, index=False)
            
            # 更新总计数
            total_successful += batch_successful
            total_failed += batch_failed
            
            # 显示批次统计
            batch_time = batch_end_time - batch_start_time
            total_processed = total_successful + total_failed
            success_rate = (total_successful / total_processed * 100) if total_processed > 0 else 0
            
            # 批次间稍作休息，避免API限流
            if batch_idx < total_batches - 1:  # 不是最后一批
                time.sleep(REQUEST_DELAY)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== 翻译完成 ===")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"总翻译条数: {len(texts_and_langs_with_idx)} 条")
    print(f"成功翻译: {total_successful} 条")
    print(f"翻译失败: {total_failed} 条")
    print(f"成功率: {(total_successful/len(texts_and_langs_with_idx)*100):.1f}%")
    print(f"平均每条翻译耗时: {elapsed_time/len(texts_and_langs_with_idx):.2f} 秒")
    print(f"最终结果已保存到: {gemini_output_path}")

def precompute_bleu2_cache():
    """预计算BLEU2分数缓存"""
    print("开始预计算BLEU2分数缓存...")
    
    # 读取数据
    if not os.path.exists(input_path):
        print(f"开发数据文件不存在: {input_path}")
        return
    
    if not os.path.exists(shot_path):
        print(f"训练数据文件不存在: {shot_path}")
        return
    
    dev_data = pd.read_csv(input_path)
    train_data = pd.read_csv(shot_path)
    
    print(f"开发数据: {len(dev_data)} 条")
    print(f"训练数据: {len(train_data)} 条")
    
    # 初始化BLEU2缓存
    bleu2_cache = {}
    
    # 使用多进程计算BLEU2分数
    def calculate_bleu2_for_dev_sample(args):
        dev_idx, dev_row, train_data_subset = args
        dev_chinese = dev_row.iloc[4]  # 中文列
        
        scores = {}
        for train_idx, train_row in train_data_subset.iterrows():
            train_chinese = train_row['中文']
            if pd.notna(train_chinese) and pd.notna(dev_chinese):
                bleu_score = calculate_bleu2_similarity(dev_chinese, train_chinese, "中文")
                scores[train_idx] = bleu_score
        
        return dev_idx, scores
    
    # 准备参数
    args_list = []
    for dev_idx, dev_row in dev_data.iterrows():
        args_list.append((dev_idx, dev_row, train_data))
    
    print(f"使用 {BLEU2_WORKERS} 个进程计算BLEU2分数...")
    
    # 使用进程池计算
    with ProcessPoolExecutor(max_workers=BLEU2_WORKERS) as executor:
        with tqdm(total=len(args_list), desc="计算BLEU2分数") as pbar:
            for dev_idx, scores in executor.map(calculate_bleu2_for_dev_sample, args_list):
                bleu2_cache[dev_idx] = scores
                pbar.update(1)
    
    # 保存缓存
    save_bleu2_cache(bleu2_cache)
    print("BLEU2分数缓存预计算完成！")

def precompute_similarity_ranking():
    """预计算相似度排序缓存"""
    print("开始预计算相似度排序缓存...")
    
    # 加载BLEU2缓存
    bleu2_cache = load_bleu2_cache()
    if bleu2_cache is None:
        print("请先运行BLEU2预计算: python infer_by_gemini.py --precompute-bleu2")
        return
    
    # 读取训练数据
    if not os.path.exists(shot_path):
        print(f"训练数据文件不存在: {shot_path}")
        return
    
    train_data = pd.read_csv(shot_path)
    print(f"训练数据: {len(train_data)} 条")
    
    # 获取所有语言
    languages = train_data['语言'].unique()
    print(f"支持的语言: {list(languages)}")
    
    # 初始化相似度排序缓存
    similarity_ranking_cache = {}
    
    print("基于BLEU2分数计算相似度排序...")
    
    for dev_idx, train_scores in tqdm(bleu2_cache.items(), desc="计算相似度排序"):
        similarity_ranking_cache[dev_idx] = {}
        
        # 为每种语言计算排序
        for lang in languages:
            # 获取该语言的训练数据索引
            lang_train_indices = train_data[
                (train_data['语言'] == lang) &
                (train_data['文本'].notna()) &
                (train_data['文本'] != "") &
                (train_data['中文'].notna()) &
                (train_data['中文'] != "")
            ].index.tolist()
            
            # 获取该语言训练样本的BLEU2分数
            lang_scores = []
            for train_idx in lang_train_indices:
                if train_idx in train_scores:
                    bleu_score = train_scores[train_idx]
                    lang_scores.append((bleu_score, train_idx))
            
            # 按BLEU2分数降序排序
            lang_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 保存排序后的训练索引
            similarity_ranking_cache[dev_idx][lang] = [train_idx for _, train_idx in lang_scores]
    
    # 保存缓存
    save_similarity_ranking_cache(similarity_ranking_cache)
    print("相似度排序缓存预计算完成！")

def test_translation(n_shot=0):
    """测试翻译功能"""
    print("测试Gemini翻译...")
    if n_shot > 0:
        print(f"使用 {n_shot}-shot 学习模式进行测试")
    
    # 检查API密钥
    if API_KEYS[0] == "your-api-key-1":
        print("请先设置Gemini API密钥！")
        return
    
    test_cases = [
        ("你好，世界！", "中文", "英语"),
        ("这是一个测试句子。", "中文", "马来语"),
        ("今天天气很好。", "中文", "泰语")
    ]
    
    # 创建翻译器以便获取完整输入
    translator = GeminiTranslator(API_KEYS, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    for i, (text, source_lang, target_lang) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试案例 {i}")
        print(f"{'='*80}")
        print(f"原文 ({source_lang}): {text}")
        print(f"目标语言: {target_lang}")
        
        # 获取完整的系统提示词
        system_prompt = translator.create_system_prompt(source_lang, target_lang, text)
        user_prompt = translator.create_translation_prompt(text)
        
        print(f"\n{'*'*60}")
        print("完整输入内容:")
        print(f"{'*'*60}")
        print("\n【系统提示词】:")
        print("-" * 40)
        print(system_prompt)
        print("-" * 40)
        
        print("\n【用户输入】:")
        print("-" * 40)
        print(user_prompt)
        print("-" * 40)
        
        # 进行翻译
        print(f"\n【翻译结果】:")
        print("-" * 40)
        result = translator.translate_text(text, source_lang, target_lang)
        print(f"译文 ({target_lang}): {result}")
        print("-" * 40)

def diagnose_api_connection():
    """诊断API连接和配置问题"""
    print("=== Gemini API 连接诊断 ===")
    
    # 1. 检查代理设置
    print("\n1. 检查代理设置:")
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
    for var in proxy_vars:
        value = os.environ.get(var)
        print(f"   {var}: {value if value else '未设置'}")
    
    # 2. 检查API配置
    print("\n2. 检查API配置:")
    print(f"   BASE_URL: {BASE_URL}")
    print(f"   MODEL_NAME: {MODEL_NAME}")
    print(f"   API_KEYS数量: {len(API_KEYS)}")
    print(f"   第一个API_KEY前缀: {API_KEYS[0][:10]}..." if API_KEYS[0] != "your-api-key-1" else "   ⚠️ 请设置真实的API密钥!")
    
    # 3. 测试网络连通性
    print("\n3. 测试网络连通性:")
    try:
        import requests
        # 测试对Google API的连接
        test_url = "https://generativelanguage.googleapis.com"
        response = requests.get(test_url, timeout=10)
        print(f"   连接到 {test_url}: ✓ 成功 (状态码: {response.status_code})")
    except Exception as e:
        print(f"   连接到Google API: ✗ 失败 - {e}")
        print("   -> 可能的问题：网络连接问题或代理配置问题")
    
    # 4. 测试简单的API调用
    print("\n4. 测试API调用:")
    if API_KEYS[0] == "your-api-key-1":
        print("   ⚠️ 跳过API测试 - 请先设置真实的API密钥")
        return
    
    try:
        api_key = API_KEYS[0]
        client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )
        
        print("   正在发送测试请求...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个翻译助手"},
                {"role": "user", "content": "hello"}
            ],
            temperature=0.0,
            max_tokens=10,
            timeout=30
        )
        
        print("   API调用成功!")
        print(f"   响应结构:")
        print(f"     - response: {type(response)}")
        print(f"     - choices: {len(response.choices) if response.choices else 0}")
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            print(f"     - message: {type(choice.message) if hasattr(choice, 'message') else 'None'}")
            if hasattr(choice, 'message') and choice.message:
                print(f"     - content: '{choice.message.content}'")
                print(f"     - content类型: {type(choice.message.content)}")
            else:
                print("     - ⚠️ message为空或不存在")
        else:
            print("     - ⚠️ choices为空")
            
    except Exception as e:
        print(f"   API调用失败: {e}")
        error_msg = str(e).lower()
        
        if "api key" in error_msg or "authentication" in error_msg:
            print("   -> 可能的问题：API密钥无效或过期")
        elif "quota" in error_msg or "billing" in error_msg:
            print("   -> 可能的问题：API配额用完或账单问题")
        elif "timeout" in error_msg:
            print("   -> 可能的问题：网络超时")
        elif "connection" in error_msg:
            print("   -> 可能的问题：网络连接问题")
        elif "rate limit" in error_msg:
            print("   -> 可能的问题：请求频率过高")
        else:
            print("   -> 未知错误，请检查错误详情")
    
    print("\n=== 诊断完成 ===")
    print("\n建议的解决方案:")
    print("1. 确保API密钥有效且有余额")
    print("2. 检查网络连接和代理设置")
    print("3. 降低并发请求数量")
    print("4. 增加请求间隔时间")
    print("5. 如果在中国大陆，确保代理工作正常")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用Gemini API进行翻译")
    parser.add_argument("--test", action="store_true", help="运行翻译测试")
    parser.add_argument("--api-keys", nargs='+', help="Gemini API密钥列表")
    parser.add_argument("--max-concurrent", type=int, default=None, 
                       help="最大并发请求数（默认等于API key数量）")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, 
                       help="请求间隔（秒）")
    parser.add_argument("--n-shot", type=int, default=0,
                       help=f"Few-shot 示例数量 (0表示不使用few-shot，默认: {DEFAULT_N_SHOT})")
    parser.add_argument("--low-concurrency", action="store_true",
                       help="使用低并发模式（推荐在网络不稳定时使用）")
    parser.add_argument("--precompute-bleu2", action="store_true",
                       help="预计算BLEU2分数缓存（用于few-shot示例选择）")
    parser.add_argument("--precompute-ranking", action="store_true",
                       help="预计算相似度排序缓存（需要先运行--precompute-bleu2）")
    parser.add_argument("--precompute-all", action="store_true",
                       help="预计算所有缓存（BLEU2分数和相似度排序）")
    parser.add_argument("--diagnose", action="store_true",
                       help="诊断API连接和配置问题")
    
    args = parser.parse_args()
    
    # 如果提供了API密钥，则更新全局变量
    if args.api_keys:
        API_KEYS = args.api_keys
        key_manager = GeminiKeyManager(API_KEYS)
        print(f"使用提供的 {len(API_KEYS)} 个API keys")
    
    # 更新配置
    if args.low_concurrency:
        MAX_CONCURRENT_REQUESTS = min(len(API_KEYS), 5)  # 降低并发数
        REQUEST_DELAY = 1.0  # 增加延迟
        print(f"使用低并发模式：并发数={MAX_CONCURRENT_REQUESTS}，延迟=1.0秒")
    else:
        if args.max_concurrent:
            MAX_CONCURRENT_REQUESTS = min(args.max_concurrent, len(API_KEYS))
        else:
            MAX_CONCURRENT_REQUESTS = len(API_KEYS)  # 默认等于API key数量
        REQUEST_DELAY = args.delay
    
    print(f"并发线程数: {MAX_CONCURRENT_REQUESTS}")
    print(f"API keys数量: {len(API_KEYS)}")
    
    # 设置 few-shot 参数
    n_shot = args.n_shot
    
    # 处理预计算选项
    if args.precompute_all:
        print("=== 开始预计算所有缓存 ===")
        precompute_bleu2_cache()
        precompute_similarity_ranking()
        print("=== 所有缓存预计算完成 ===")
    elif args.precompute_bleu2:
        print("=== 开始预计算BLEU2分数缓存 ===")
        precompute_bleu2_cache()
        print("=== BLEU2分数缓存预计算完成 ===")
    elif args.precompute_ranking:
        print("=== 开始预计算相似度排序缓存 ===")
        precompute_similarity_ranking()
        print("=== 相似度排序缓存预计算完成 ===")
    elif args.diagnose:
        diagnose_api_connection()
    elif args.test:
        test_translation(n_shot=n_shot)
    else:
        translate_gemini(n_shot=n_shot) 