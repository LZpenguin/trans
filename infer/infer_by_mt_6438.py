import os
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
        return os.path.join(base_dir, f"output/qwen-mt/dev_plus_{n_shot}shot.csv")
    else:
        return os.path.join(base_dir, f"output/qwen-mt/result.csv")

# 语言映射 - 从中文名称到英文名称
lang_map = {
    "马来语": "Malay",
    "泰语": "Thai", 
    "英语": "English"
}

# Qwen-MT API配置
API_KEY = "	sk-aaa6ee01a3a34cc195d0acbd86c06a51"  # 请替换为您的阿里云百炼API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-mt-plus"

# 并发配置
MAX_CONCURRENT_REQUESTS = 3  # 最大并发请求数，Qwen-MT建议适当降低
REQUEST_DELAY = 0.5  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_FACTOR = 2  # 退避因子
# Few-shot 配置
DEFAULT_N_SHOT = 10  # 默认示例数量

# 性能优化配置
BLEU2_WORKERS = min(mp.cpu_count(), 96)  # BLEU2计算的并行工作进程数
BATCH_SIZE = 1000  # 批量处理大小

# 全局分词缓存
_tokenize_cache = {}

# 从原文件复制的必要函数
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

def load_bleu2_cache():
    """加载BLEU2缓存"""
    if not os.path.exists(bleu2_cache_file):
        print(f"BLEU2缓存文件不存在: {bleu2_cache_file}")
        print("请先运行预计算: python infer_by_api_v21_rag_b4_nshot.py --precompute-bleu2")
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

class QwenMTTranslator:
    def __init__(self, api_key, base_url, model_name, n_shot=0):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
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
        
    def load_translation_memory(self, target_lang, current_text=None, dev_idx=None):
        """从训练数据加载翻译记忆，基于缓存的相似度排序选择最相似的示例"""
        # 为缓存添加dev_idx来区分不同的查询
        cache_key = f"{target_lang}_{dev_idx if dev_idx is not None else hash(current_text) if current_text else 'default'}"
        
        if cache_key in self.examples_cache:
            return self.examples_cache[cache_key]
            
        tm_list = []
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
                                tm_list.append({
                                    "source": chinese_text,
                                    "target": translated_text
                                })
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
                            tm_list.append({
                                "source": chinese_text,
                                "target": translated_text
                            })
                    else:
                        # 如果没有缓存，随机选择（兼容旧逻辑）
                        sampled_data = lang_data.sample(n=self.n_shot, random_state=42)
                        for _, row in sampled_data.iterrows():
                            chinese_text = row['中文']  # 中文列
                            translated_text = row['文本']  # 翻译结果
                            tm_list.append({
                                "source": chinese_text,
                                "target": translated_text
                            })
                else:
                    # 如果数据不够，使用所有可用的
                    for _, row in lang_data.iterrows():
                        chinese_text = row['中文']
                        translated_text = row['文本']
                        tm_list.append({
                            "source": chinese_text,
                            "target": translated_text
                        })
                        
            except Exception as e:
                print(f"加载翻译记忆时出错: {e}")
                tm_list = []
        
        self.examples_cache[cache_key] = tm_list
        return tm_list
    
    def translate_text(self, text, source_lang_chinese, target_lang_chinese, dev_idx=None):
        """使用Qwen-MT API翻译文本，带重试和指数退避"""
        # 转换为英文语言名称
        source_lang = "Chinese"  # 统一从中文翻译
        target_lang = lang_map.get(target_lang_chinese, target_lang_chinese)
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 准备翻译选项
                translation_options = {
                    "source_lang": source_lang,
                    "target_lang": target_lang
                }
                
                # 如果启用翻译记忆，添加tm_list
                if self.n_shot > 0:
                    tm_list = self.load_translation_memory(target_lang_chinese, text, dev_idx)
                    if tm_list:
                        translation_options["tm_list"] = tm_list
                        # print(f"为翻译添加了 {len(tm_list)} 条翻译记忆")  # 调试时启用
                
                # 调用Qwen-MT API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                    temperature=0.0,
                    max_tokens=512,
                    stream=False,
                    timeout=30,  # 添加30秒超时
                    extra_body={
                        "translation_options": translation_options
                    }
                )
                
                translated_text = response.choices[0].message.content.strip()
                time.sleep(REQUEST_DELAY)  # 添加请求延迟
                return translated_text
                
            except Exception as e:
                print(f"翻译过程中出错: {str(e)}")
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
        if pd.isna(text) or text == "":
            return index, ""
        
        # 添加单个任务的调试信息
        try:
            result = translator.translate_text(text, source_lang, target_lang, dev_idx)
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

def translate_qwen_mt(n_shot=0):
    """主翻译函数"""
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    print("开始使用Qwen-MT API进行翻译...")
    print("提示：如果程序卡住，可以按 Ctrl+C 强制终止")
    if n_shot > 0:
        print(f"使用 {n_shot} 条翻译记忆")
    
    # 获取输出路径
    qwen_output_path = get_output_path(n_shot)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(qwen_output_path), exist_ok=True)
    
    # 检查API密钥
    if API_KEY == "sk-xxx":
        print("警告：请在脚本中设置您的阿里云百炼API密钥！")
        print("请将API_KEY变量替换为您的实际API密钥")
        return
    
    # 首先检查输出文件是否已存在
    if os.path.exists(qwen_output_path):
        print(f"发现已存在的结果文件: {qwen_output_path}")
        print("加载已有结果，将跳过已完成的翻译...")
        df = pd.read_csv(qwen_output_path)
        print(f"从结果文件读取到 {len(df)} 条数据")
        
        # 确保answer列存在
        if 'answer' not in df.columns:
            df['answer'] = ""
            print("警告：结果文件中没有'answer'列，已添加空列")
    else:
        print("未发现已存在的结果文件，从输入文件开始...")
        # 读取ASR结果文件
        if not os.path.exists(input_path):
            print(f"输入文件不存在: {input_path}")
            print("请先运行ASR处理生成输入文件")
            return
        
        df = pd.read_csv(input_path)
        print(f"从输入文件读取到 {len(df)} 条数据")
        
        # 添加answer列
        df['answer'] = ""
    
    # 获取需要翻译的数据
    need_translation = df[df['answer'].isna() | (df['answer'] == "")]
    already_translated = len(df) - len(need_translation)
    
    print(f"数据统计：")
    print(f"  - 总计: {len(df)} 条")
    print(f"  - 已翻译: {already_translated} 条")
    print(f"  - 需要翻译: {len(need_translation)} 条")
    
    if len(need_translation) == 0:
        print("✅ 所有数据已翻译完成！")
        return
    
    # 准备翻译数据（包含dev_idx）
    texts_and_langs_with_idx = []
    indices = []
    
    for idx, row in need_translation.iterrows():
        chinese_text = row.iloc[4]  # 中文列
        target_lang_chinese = row.iloc[3]  # 语言列
        
        texts_and_langs_with_idx.append((chinese_text, "中文", target_lang_chinese, idx))  # 添加dev_idx
        indices.append(idx)
    
    # 创建Qwen-MT翻译器
    translator = QwenMTTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    # 分批翻译，每批保存一次
    batch_size = MAX_CONCURRENT_REQUESTS
    total_batches = (len(texts_and_langs_with_idx) + batch_size - 1) // batch_size
    
    print(f"开始翻译，共 {total_batches} 批，每批 {batch_size} 条...")
    print(f"使用 {MAX_CONCURRENT_REQUESTS} 个并发线程...")
    
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
            df.to_csv(qwen_output_path, index=False)
            
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
    print(f"本轮翻译条数: {len(texts_and_langs_with_idx)} 条")
    print(f"本轮成功翻译: {total_successful} 条")
    print(f"本轮翻译失败: {total_failed} 条")
    print(f"本轮成功率: {(total_successful/len(texts_and_langs_with_idx)*100):.1f}%")
    if len(texts_and_langs_with_idx) > 0:
        print(f"平均每条翻译耗时: {elapsed_time/len(texts_and_langs_with_idx):.2f} 秒")
    
    # 显示最终统计
    final_translated = len(df[df['answer'].notna() & (df['answer'] != "")])
    final_total = len(df)
    print(f"\n=== 最终统计 ===")
    print(f"总数据量: {final_total} 条")
    print(f"已完成翻译: {final_translated} 条")
    print(f"整体完成率: {(final_translated/final_total*100):.1f}%")
    print(f"最终结果已保存到: {qwen_output_path}")

def test_translation(n_shot=0):
    """测试翻译功能"""
    print("测试Qwen-MT翻译...")
    if n_shot > 0:
        print(f"使用 {n_shot} 条翻译记忆进行测试")
    
    # 检查API密钥
    if API_KEY == "sk-xxx":
        print("请先设置阿里云百炼API密钥！")
        return
    
    test_cases = [
        ("你好，世界！", "中文", "英语"),
        ("这是一个测试句子。", "中文", "马来语"),
        ("今天天气很好。", "中文", "泰语")
    ]
    
    # 创建翻译器
    translator = QwenMTTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    for i, (text, source_lang, target_lang) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试案例 {i}")
        print(f"{'='*80}")
        print(f"原文 ({source_lang}): {text}")
        print(f"目标语言: {target_lang}")
        
        # 如果启用翻译记忆，显示将要使用的翻译记忆
        if n_shot > 0:
            tm_list = translator.load_translation_memory(target_lang, text)
            print(f"\n【翻译记忆】({len(tm_list)} 条):")
            print("-" * 40)
            for j, tm in enumerate(tm_list, 1):
                print(f"{j}. 源: {tm['source']}")
                print(f"   译: {tm['target']}")
                print()
            print("-" * 40)
        
        # 进行翻译
        print(f"\n【翻译结果】:")
        print("-" * 40)
        result = translator.translate_text(text, source_lang, target_lang)
        print(f"译文 ({target_lang}): {result}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用Qwen-MT API进行翻译")
    parser.add_argument("--test", action="store_true", help="运行翻译测试")
    parser.add_argument("--api-key", type=str, help="阿里云百炼API密钥")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, 
                       help="最大并发请求数")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, 
                       help="请求间隔（秒）")
    parser.add_argument("--n-shot", type=int, default=0,
                       help=f"翻译记忆条数 (0表示不使用翻译记忆，默认: {DEFAULT_N_SHOT})")
    parser.add_argument("--low-concurrency", action="store_true",
                       help="使用低并发模式（推荐在网络不稳定时使用）")
    
    args = parser.parse_args()
    
    # 如果提供了API密钥，则更新全局变量
    if args.api_key:
        API_KEY = args.api_key
    
    # 更新配置
    if args.low_concurrency:
        MAX_CONCURRENT_REQUESTS = 20  # 降低并发数
        REQUEST_DELAY = 0.5  # 增加延迟
        print("使用低并发模式：并发数=20，延迟=0.5秒")
    else:
        MAX_CONCURRENT_REQUESTS = args.max_concurrent
        REQUEST_DELAY = args.delay
    
    # 设置翻译记忆参数
    n_shot = args.n_shot
    
    if args.test:
        test_translation(n_shot=n_shot)
    else:
        translate_qwen_mt(n_shot=n_shot)
