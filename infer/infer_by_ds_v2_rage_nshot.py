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

# 延迟导入embedding模块，避免依赖问题
TextEmbeddingFAISS = None

def import_text_embedding():
    """延迟导入TextEmbeddingFAISS模块"""
    global TextEmbeddingFAISS
    if TextEmbeddingFAISS is None:
        try:
            from text_embedding import TextEmbeddingFAISS
            return TextEmbeddingFAISS
        except ImportError as e:
            print(f"无法导入TextEmbeddingFAISS模块: {e}")
            print("请安装所需依赖: pip install torch transformers faiss-cpu")
            raise e
    return TextEmbeddingFAISS

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

# 移除BLEU2相关的导入，现在使用embedding

dev_mode = True

# 路径配置
base_dir = os.getcwd()
shot_path = os.path.join(base_dir, "data/text_data/train.csv")
if dev_mode:
    input_path = os.path.join(base_dir, "data/text_data/dev.csv")
else:
    input_path = os.path.join(base_dir, "output/testa_asr.csv")

# 保留旧缓存目录配置用于兼容性检查
cache_dir = os.path.join(base_dir, f"output/cache{'_dev' if 'dev' in input_path else ''}")
bleu2_cache_file = os.path.join(cache_dir, "bleu2_cache.json")  # 仅用于兼容性检查

# 新的embedding缓存配置，根据dev_mode区分缓存目录
embedding_cache_dir = os.path.join(base_dir, f"output/cache_embedding{'_dev' if dev_mode else ''}")

def get_output_path(n_shot):
    if dev_mode:
        return os.path.join(base_dir, f"output/deepseek/dev_p2_t0_rag_{n_shot}shot.csv")
    else:
        return os.path.join(base_dir, f"output/deepseek/result.csv")


# 语言映射
lang_map = {
    "马来语": "Malay",
    "泰语": "Thai", 
    "英语": "English"
}

# DeepSeek API配置
API_KEY = "sk-70a4f9feb82a4de4aa64fb5913314666"  # 请替换为您的DeepSeek API Key
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 并发配置
MAX_CONCURRENT_REQUESTS = 100  # 最大并发请求数
REQUEST_DELAY = 0.5  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_FACTOR = 2  # 退避因子
# Few-shot 配置
DEFAULT_N_SHOT = 3  # 默认示例数量


# 移除了旧的BLEU2相似度排序缓存函数，现在使用embedding

# 移除了所有BLEU2相关的分词和计算函数，现在使用embedding

# 移除了BLEU2批量计算函数

def precompute_bleu2_cache_optimized():
    """已废弃：BLEU2预计算函数，请使用embedding"""
    print("错误：此功能已废弃，请使用 --precompute-embedding")
    return

# 移除了大型BLEU2预计算函数实现

def precompute_bleu2_cache_simple():
    """已废弃：BLEU2预计算函数，请使用embedding"""
    print("错误：此功能已废弃，请使用 --precompute-embedding")
    return

# 移除了简化版BLEU2预计算函数实现

def check_cache_status():
    """检查缓存状态"""
    print("=== 缓存状态检查 ===")
    
    # 检查新的FAISS embedding缓存
    print("\n-- FAISS Embedding缓存 --")
    try:
        TextEmbeddingClass = import_text_embedding()
        text_embedder = TextEmbeddingClass(cache_dir=embedding_cache_dir)
        text_embedder.get_statistics()
    except ImportError:
        print("✗ TextEmbeddingFAISS模块未安装，跳过embedding缓存检查")
    except Exception as e:
        print(f"初始化embedding模块失败: {e}")
    
    # 检查训练数据
    print("\n-- 数据文件 --")
    if os.path.exists(shot_path):
        try:
            train_df = pd.read_csv(shot_path)
            print(f"✓ 训练数据文件存在: {shot_path} ({len(train_df)} 条)")
        except Exception as e:
            print(f"✗ 训练数据文件读取失败: {e}")
    else:
        print(f"✗ 训练数据文件不存在: {shot_path}")
    
    # 检查开发数据
    if os.path.exists(input_path):
        try:
            dev_df = pd.read_csv(input_path)
            print(f"✓ 开发数据文件存在: {input_path} ({len(dev_df)} 条)")
        except Exception as e:
            print(f"✗ 开发数据文件读取失败: {e}")
    else:
        print(f"✗ 开发数据文件不存在: {input_path}")
    
    # 检查模型文件
    print("\n-- 模型文件 --")
    model_path = "/home/zbtrs/lz/trans/trans/models/Qwen3-Embedding-0.6B"
    if os.path.exists(model_path):
        print(f"✓ Qwen3-Embedding模型存在: {model_path}")
    else:
        print(f"✗ Qwen3-Embedding模型不存在: {model_path}")
        print("  请确保模型文件已正确下载")

def generate_similarity_ranking_cache():
    """已废弃的函数，请使用embedding相似度"""
    print("错误：此功能已废弃，请使用 --precompute-embedding")
    return
    
def precompute_embedding_similarity():
    """使用embedding预计算并添加训练数据到向量数据库"""
    print("开始使用embedding预计算相似度排序...")
    
    # 读取数据
    if not os.path.exists(shot_path):
        print(f"训练数据文件不存在: {shot_path}")
        return
    
    print("正在读取数据文件...")
    train_df = pd.read_csv(shot_path)
    
    print(f"训练数据: {len(train_df)} 条")
    
    # 筛选有效的训练数据
    train_df_valid = train_df[
        (train_df['语言'].notna()) &
        (train_df['文本'].notna()) &
        (train_df['文本'] != "") &
        (train_df['中文'].notna()) &
        (train_df['中文'] != "")
    ].copy()
    
    print(f"有效训练数据: {len(train_df_valid)} 条")
    
    # 初始化embedding模块
    print("初始化embedding模块...")
    TextEmbeddingClass = import_text_embedding()
    text_embedder = TextEmbeddingClass(cache_dir=embedding_cache_dir)
    
    # 批量添加训练数据中文文本到向量数据库
    chinese_texts = train_df_valid['中文'].tolist()
    print(f"正在添加 {len(chinese_texts)} 条中文文本到向量数据库...")
    text_embedder.add_texts(chinese_texts)
    
    print("embedding预计算完成！")
    print("现在可以使用向量数据库进行快速相似度搜索了。")

# 移除了废弃的BLEU2相关函数

class DeepSeekTranslator:
    def __init__(self, api_key, base_url, model_name, n_shot=0):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.n_shot = n_shot
        self.examples_cache = {}  # 缓存示例数据
        self.train_data = None  # 训练数据缓存
        
        # 初始化embedding模块
        self.text_embedder = None
        if self.n_shot > 0:
            print("初始化文本embedding模块...")
            TextEmbeddingClass = import_text_embedding()
            self.text_embedder = TextEmbeddingClass(cache_dir=embedding_cache_dir)
            
            # 预加载训练数据到内存
            if os.path.exists(shot_path):
                print("加载训练数据到内存...")
                self.train_data = pd.read_csv(shot_path)
                print(f"训练数据加载完成：{len(self.train_data)} 条")
            else:
                print(f"警告：训练数据文件不存在: {shot_path}")
        
    def load_examples(self, target_lang, current_text=None, dev_idx=None):
        """从训练数据加载指定语言的翻译示例，基于embedding相似度选择最相似的示例"""
        # 为缓存添加唯一标识
        cache_key = f"{target_lang}_{hash(current_text) if current_text else 'random'}"
        
        if cache_key in self.examples_cache:
            return self.examples_cache[cache_key]
            
        examples = []
        if self.n_shot > 0 and self.train_data is not None and self.text_embedder is not None:
            try:
                # 筛选出目标语言的数据，且有有效翻译结果的
                lang_data = self.train_data[
                    (self.train_data['语言'] == target_lang) &  # 语言列匹配
                    (self.train_data['文本'].notna()) &  # 有翻译结果
                    (self.train_data['文本'] != "") &  # 翻译结果非空
                    (self.train_data['中文'].notna()) &  # 中文文本非空
                    (self.train_data['中文'] != "")
                ]
                
                if len(lang_data) >= self.n_shot and current_text:
                    # 使用embedding搜索最相似的文本，增加搜索范围
                    similar_results = self.text_embedder.search_similar(current_text, top_k=self.n_shot * 3)
                    
                    # 添加调试信息
                    print(f"DEBUG: 搜索到 {len(similar_results)} 个相似结果")
                    if similar_results:
                        print(f"DEBUG: 相似度范围 [{min(s[1] for s in similar_results):.4f}, {max(s[1] for s in similar_results):.4f}]")
                    
                    # 从相似结果中筛选目标语言的样本
                    matched_count = 0
                    for similar_text, score in similar_results:
                        # 在lang_data中查找匹配的文本
                        matching_rows = lang_data[lang_data['中文'] == similar_text]
                        for _, row in matching_rows.iterrows():
                            if matched_count < self.n_shot:
                                chinese_text = row['中文']
                                translated_text = row['文本']
                                examples.append((chinese_text, translated_text, score))  # 保存相似度分数
                                matched_count += 1
                            else:
                                break
                        
                        if matched_count >= self.n_shot:
                            break
                    
                    # 如果找到的样本不够，随机补充
                    if len(examples) < self.n_shot:
                        remaining_data = lang_data[~lang_data['中文'].isin([ex[0] for ex in examples])]
                        if len(remaining_data) > 0:
                            needed = self.n_shot - len(examples)
                            sampled_data = remaining_data.sample(n=min(needed, len(remaining_data)), random_state=42)
                            for _, row in sampled_data.iterrows():
                                chinese_text = row['中文']
                                translated_text = row['文本']
                                examples.append((chinese_text, translated_text, 0.0))  # 随机样本相似度设为0
                
                elif len(lang_data) > 0:
                    # 如果没有current_text或数据不够，随机选择
                    sample_size = min(self.n_shot, len(lang_data))
                    sampled_data = lang_data.sample(n=sample_size, random_state=42)
                    for _, row in sampled_data.iterrows():
                        chinese_text = row['中文']
                        translated_text = row['文本']
                        examples.append((chinese_text, translated_text, 0.0))  # 随机样本相似度设为0
                        
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
            return system_prompt, []
        else:
            # 包含示例的系统提示词，基于embedding相似度选择
            examples = self.load_examples(target_lang, current_text, dev_idx)
            
            if examples:
                examples_text = "\n\n以下是一些翻译示例供参考（按相似度排序）：\n"
                example_details = []  # 存储示例详情用于测试打印
                
                for i, (chinese, translated, score) in enumerate(examples, 1):
                    examples_text += f"示例{i}:\n"
                    examples_text += f"中文: {chinese}\n"
                    examples_text += f"{target_lang}: {translated}\n"
                    examples_text += f"相似度: {score:.4f}\n\n"
                    
                    # 使用已保存的相似度分数
                    example_details.append({
                        'chinese': chinese,
                        'translated': translated,
                        'similarity': score
                    })
                
                system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。{examples_text}请参考以上示例的翻译风格和质量，将用户输入的中文文本翻译成{target_lang}。请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
                return system_prompt, example_details
            else:
                # 如果没有找到示例，使用原有提示词
                system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。用户将输入中文文本，请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
                return system_prompt, []
    
    def create_translation_prompt(self, text):
        """创建翻译提示词"""
        return text
    
    def translate_text(self, text, source_lang, target_lang, dev_idx=None):
        """使用DeepSeek API翻译文本，带重试和指数退避"""
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 创建系统和用户提示，传入dev_idx用于示例选择
                system_prompt, example_details = self.create_system_prompt(source_lang, target_lang, text, dev_idx)
                user_prompt = self.create_translation_prompt(text)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=128,
                    stream=False,
                    timeout=30  # 添加30秒超时
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

def translate_deepseek(n_shot=0):
    """主翻译函数"""
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    print("开始使用DeepSeek API进行翻译...")
    print("提示：如果程序卡住，可以按 Ctrl+C 强制终止")
    if n_shot > 0:
        print(f"使用 {n_shot}-shot 学习模式")
    
    # 获取输出路径
    deepseek_output_path = get_output_path(n_shot)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(deepseek_output_path), exist_ok=True)
    
    # 检查API密钥
    if API_KEY == "<DeepSeek API Key>":
        print("警告：请在脚本中设置您的DeepSeek API密钥！")
        print("请将API_KEY变量替换为您的实际API密钥")
        return
    
    # 读取ASR结果文件
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        print("请先运行ASR处理生成输入文件")
        return
    
    df = pd.read_csv(input_path)
    print(f"读取到 {len(df)} 条数据")
    
    # 检查输出文件是否已存在
    existing_results = {}
    if os.path.exists(deepseek_output_path):
        print(f"发现已存在的结果文件: {deepseek_output_path}")
        try:
            existing_df = pd.read_csv(deepseek_output_path)
            print(f"加载现有结果文件，包含 {len(existing_df)} 条数据")
            
            # 如果现有结果文件有 answer 列，将其映射到索引
            if 'answer' in existing_df.columns:
                existing_results = existing_df['answer'].to_dict()
                completed_count = sum(1 for v in existing_results.values() 
                                    if pd.notna(v) and str(v).strip() != "")
                print(f"现有结果中已完成翻译: {completed_count} 条")
            else:
                print("现有结果文件中没有找到 answer 列")
        except Exception as e:
            print(f"加载现有结果文件失败: {e}")
            print("将重新开始翻译")
    else:
        print(f"结果文件不存在: {deepseek_output_path}")
        print("将创建新的结果文件")
    
    # 初始化answer列，优先使用现有结果
    if 'answer' not in df.columns:
        df['answer'] = ""
    
    # 将现有结果合并到数据框中
    if existing_results:
        for idx in existing_results:
            if idx < len(df) and pd.notna(existing_results[idx]) and str(existing_results[idx]).strip() != "":
                df.loc[idx, 'answer'] = existing_results[idx]
    
    # 统计已有结果
    already_completed = df[df['answer'].notna() & (df['answer'] != "")]
    print(f"已有翻译结果: {len(already_completed)} 条")
    
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
    
    # 创建DeepSeek翻译器
    translator = DeepSeekTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    # 分批翻译，每10条保存一次
    batch_size = MAX_CONCURRENT_REQUESTS  # 每批处理10条
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
            df.to_csv(deepseek_output_path, index=False)
            
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
    
    # 统计最终结果
    final_completed = df[df['answer'].notna() & (df['answer'] != "")]
    
    print(f"\n=== 翻译完成 ===")
    print(f"总数据条数: {len(df)} 条")
    print(f"之前已完成: {len(already_completed)} 条")
    print(f"本次需要翻译: {len(texts_and_langs_with_idx)} 条")
    print(f"本次成功翻译: {total_successful} 条")
    print(f"本次翻译失败: {total_failed} 条")
    print(f"最终完成总数: {len(final_completed)} 条")
    print(f"总体完成率: {(len(final_completed)/len(df)*100):.1f}%")
    if len(texts_and_langs_with_idx) > 0:
        print(f"本次成功率: {(total_successful/len(texts_and_langs_with_idx)*100):.1f}%")
        print(f"本次翻译耗时: {elapsed_time:.2f} 秒")
        print(f"平均每条翻译耗时: {elapsed_time/len(texts_and_langs_with_idx):.2f} 秒")
    print(f"最终结果已保存到: {deepseek_output_path}")

def sync_translate_one(text, source_lang, target_lang, n_shot=0):
    """同步翻译单个文本（用于测试）"""
    translator = DeepSeekTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    return translator.translate_text(text, source_lang, target_lang)

def test_embedding():
    """测试embedding功能"""
    print("测试Embedding功能...")
    
    # 初始化embedding模块
    TextEmbeddingClass = import_text_embedding()
    text_embedder = TextEmbeddingClass(cache_dir=embedding_cache_dir)
    
    # 测试文本
    test_texts = [
        "你好，世界！",
        "这是一个测试句子。",
        "今天天气很好。",
        "我喜欢编程和学习新技术。",
        "程序开发是一项有趣的工作",
        "软件工程师的日常工作。"
    ]
    
    print("\n=== 测试批量添加文本 ===")
    text_embedder.add_texts(test_texts)
    
    print("\n=== 测试相似度搜索 ===")
    queries = [
        "编程和开发",
        "天气很棒",
        "打招呼"
    ]
    
    for query in queries:
        print(f"\n查询: '{query}'")
        similar_results = text_embedder.search_similar(query, top_k=3)
        print("最相似的文本:")
        for i, (text, score) in enumerate(similar_results, 1):
            print(f"  {i}. 相似度 {score:.4f}: {text}")
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    text_embedder.get_statistics()

def test_translation(n_shot=0):
    """测试翻译功能 - 从dev.csv抽取数据进行测试"""
    print("测试DeepSeek翻译...")
    if n_shot > 0:
        print(f"使用 {n_shot}-shot 学习模式进行测试")
    
    # 读取dev.csv数据
    if not os.path.exists(input_path):
        print(f"开发数据文件不存在: {input_path}")
        return
    
    print(f"正在读取开发数据: {input_path}")
    dev_df = pd.read_csv(input_path)
    print(f"开发数据总数: {len(dev_df)} 条")
    
    # 筛选有效数据并按语言分组
    valid_data = dev_df[
        (dev_df.iloc[:, 3].notna()) &  # 语言列
        (dev_df.iloc[:, 4].notna()) &  # 中文列
        (dev_df.iloc[:, 4] != "") &    # 中文非空
        (dev_df.iloc[:, 5].notna()) &  # 参考翻译列
        (dev_df.iloc[:, 5] != "")      # 参考翻译非空
    ]
    
    print(f"有效数据: {len(valid_data)} 条")
    
    # 按语言分组并每个语言选择几个样本
    languages = valid_data.iloc[:, 3].unique()
    print(f"包含语言: {list(languages)}")
    
    # 每种语言选择2个样本进行测试
    test_samples = []
    for lang in languages:
        lang_data = valid_data[valid_data.iloc[:, 3] == lang]
        # 随机选择2个样本
        sampled = lang_data.sample(n=min(2, len(lang_data)), random_state=42)
        for idx, row in sampled.iterrows():
            test_samples.append((idx, row))
    
    print(f"选择了 {len(test_samples)} 个测试样本")
    
    # 创建翻译器
    translator = DeepSeekTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    # 测试每个样本
    for i, (dev_idx, row) in enumerate(test_samples, 1):
        chinese_text = row.iloc[4]      # 中文文本
        target_lang = row.iloc[3]       # 目标语言
        reference_translation = row.iloc[2]  # 参考翻译（文本列）
        
        print(f"\n{'='*100}")
        print(f"测试样本 {i}/{len(test_samples)} (dev_idx={dev_idx})")
        print(f"{'='*100}")
        print(f"目标语言: {target_lang}")
        print(f"中文原文: {chinese_text}")
        print(f"参考翻译: {reference_translation}")
        
        # 获取系统提示词和示例详情
        system_prompt, example_details = translator.create_system_prompt("中文", target_lang, chinese_text, dev_idx)
        user_prompt = translator.create_translation_prompt(chinese_text)
        
        # 显示Few-shot示例信息
        if example_details:
            print(f"\n【Few-shot 示例信息】 (共 {len(example_details)} 个)")
            print("-" * 80)
            for j, example in enumerate(example_details, 1):
                similarity_info = f"相似度: {example['similarity']:.4f}" if example['similarity'] is not None else "相似度: 未计算"
                print(f"示例{j} ({similarity_info})")
                print(f"  中文: {example['chinese']}")
                print(f"  {target_lang}: {example['translated']}")
                print()
        else:
            print(f"\n【无Few-shot示例】 (n_shot={n_shot})")
        
        # 显示完整的API输入
        print(f"\n【API输入内容】")
        print("-" * 80)
        print("系统提示词:")
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        print(f"\n用户输入: {user_prompt}")
        
        # 进行翻译
        print(f"\n【翻译执行】")
        print("-" * 80)
        
        try:
            start_time = time.time()
            translation_result = translator.translate_text(chinese_text, "中文", target_lang, dev_idx)
            end_time = time.time()
            
            print(f"翻译耗时: {end_time - start_time:.2f} 秒")
            
            # 显示结果对比
            print(f"\n【结果对比】")
            print("-" * 80)
            print(f"模型翻译: {translation_result}")
            print(f"参考翻译: {reference_translation}")
            
            # 简单的结果评估
            if translation_result.strip():
                print("✓ 翻译成功")
                if translation_result.strip().lower() == reference_translation.strip().lower():
                    print("✓ 与参考翻译完全匹配")
                else:
                    print("◯ 与参考翻译不同")
            else:
                print("✗ 翻译失败 (返回空结果)")
                
        except Exception as e:
            print(f"✗ 翻译出错: {e}")
        
        # 测试间隔
        if i < len(test_samples):
            print(f"\n等待 {REQUEST_DELAY} 秒后继续下一个测试...")
            time.sleep(REQUEST_DELAY)
    
    print(f"\n{'='*100}")
    print("所有测试完成！")
    print(f"{'='*100}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用DeepSeek API进行翻译")
    parser.add_argument("--test", action="store_true", help="运行翻译测试")
    parser.add_argument("--test-embedding", action="store_true", help="测试embedding功能")
    parser.add_argument("--api-key", type=str, help="DeepSeek API密钥")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, 
                       help="最大并发请求数")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, 
                       help="请求间隔（秒）")
    parser.add_argument("--n-shot", type=int, default=5,
                       help=f"Few-shot 示例数量 (0表示不使用few-shot，默认: {DEFAULT_N_SHOT})")
    # 移除了所有BLEU2相关的命令行参数
    parser.add_argument("--precompute-embedding", action="store_true",
                       help="使用Qwen3-Embedding预计算文本相似度排序缓存（推荐）")
    parser.add_argument("--check-cache", action="store_true",
                       help="检查缓存状态")
    parser.add_argument("--low-concurrency", action="store_true",
                       help="使用低并发模式（推荐在网络不稳定时使用）")
    # 移除了BLEU2_WORKERS参数
    
    args = parser.parse_args()
    
    # 如果提供了API密钥，则更新全局变量
    if args.api_key:
        API_KEY = args.api_key
    
    # 更新配置
    if args.low_concurrency:
        MAX_CONCURRENT_REQUESTS = 10  # 降低并发数
        REQUEST_DELAY = 1.0  # 增加延迟
        print("使用低并发模式：并发数=10，延迟=1.0秒")
    else:
        MAX_CONCURRENT_REQUESTS = args.max_concurrent
        REQUEST_DELAY = args.delay
    # 设置 few-shot 参数
    n_shot = args.n_shot
    
    # 执行任务
    if args.precompute_embedding:
        precompute_embedding_similarity()
    elif args.check_cache:
        check_cache_status()
    elif args.test:
        test_translation(n_shot=n_shot)
    elif args.test_embedding:
        test_embedding()
    else:
        translate_deepseek(n_shot=n_shot)