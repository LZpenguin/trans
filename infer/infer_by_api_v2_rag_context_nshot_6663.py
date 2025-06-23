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
        return os.path.join(base_dir, f"output/deepseek/dev_p2_t0_rag_context_{n_shot}shot.csv")
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

# 性能优化配置
BLEU2_WORKERS = min(mp.cpu_count(), 96)  # BLEU2计算的并行工作进程数
BATCH_SIZE = 1000  # 批量处理大小

# 全局分词缓存
_tokenize_cache = {}


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

def tokenize_for_bleu(text, language):
    """根据语言类型进行适当的分词（中文使用字符级分词）"""
    if language in ["Thai", "泰语"]:
        if THAI_TOKENIZER_AVAILABLE:
            # 使用泰语分词工具
            try:
                tokens = thai_tokenize(text, engine='newmm')
                return tokens
            except Exception as e:
                # 如果分词失败，使用字符级分词（去除空格）
                return list(text.replace(" ", ""))
        else:
            # 如果没有安装pythainlp，使用字符级分词
            return list(text.replace(" ", ""))
    elif language in ["English", "英语"]:
        # 英语使用空格分词
        return text.split()
    elif language in ["Malay", "马来语"]:
        # 马来语也使用空格分词
        return text.split()
    elif language in ["中文", "Chinese"]:
        # 中文使用字符级分词
        return list(text.replace(" ", ""))
    else:
        # 默认使用空格分词
        return text.split()

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

# 独立的工作函数，避免序列化问题
def mp_tokenize_for_bleu(text, language):
    """多进程安全的分词函数"""
    global THAI_AVAILABLE
    if language in ["Thai", "泰语"]:
        if THAI_AVAILABLE:
            try:
                from pythainlp import word_tokenize as thai_tokenize
                return thai_tokenize(text, engine='newmm')
            except Exception as e:
                return list(text.replace(" ", ""))
        else:
            return list(text.replace(" ", ""))
    elif language in ["English", "英语"]:
        return text.split()
    elif language in ["Malay", "马来语"]:
        return text.split()
    elif language in ["中文", "Chinese"]:
        return list(text.replace(" ", ""))
    else:
        return text.split()

def mp_calculate_bleu2_fast(tokens1, tokens2):
    """多进程安全的BLEU2计算函数"""
    try:
        if len(tokens1) > 0 and len(tokens2) > 0:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([tokens1], tokens2, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            return bleu_score
        else:
            return 0.0
    except Exception as e:
        return 0.0

def calculate_bleu2_batch_worker(args):
    """批量计算BLEU2分数的工作函数（多进程版本）"""
    import os
    import sys
    
    dev_idx, dev_text, train_data_batch, language = args
    worker_pid = os.getpid()
    
    try:
        # 对dev文本分词（只需要分一次）
        dev_tokens = mp_tokenize_for_bleu(dev_text, language)
        results = {}
        
        for i, (train_idx, train_text) in enumerate(train_data_batch):
            # 对train文本分词
            train_tokens = mp_tokenize_for_bleu(train_text, language)
            # 计算BLEU2分数
            bleu_score = mp_calculate_bleu2_fast(dev_tokens, train_tokens)
            results[train_idx] = bleu_score
            
            # 每处理500个样本打印一次进度（减少输出频率）
            # if i % 500 == 0 and i > 0:
                # print(f"    Worker-{worker_pid}: dev_idx={dev_idx}, {i}/{len(train_data_batch)} done")
        
        return dev_idx, results, len(train_data_batch)
    except Exception as e:
        # print(f"Worker-{worker_pid} error dev_idx={dev_idx}: {e}")
        import traceback
        traceback.print_exc()
        return dev_idx, {}, 0

def precompute_bleu2_cache_optimized():
    """优化版本：使用并行计算和批量处理预计算BLEU2分数"""
    print("开始预计算BLEU2相似度分数（优化版本）...")
    
    # 设置多进程环境
    setup_multiprocessing()
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    
    # 如果缓存文件已存在，询问是否重新计算
    if os.path.exists(bleu2_cache_file):
        response = input(f"BLEU2缓存文件已存在: {bleu2_cache_file}\n是否重新计算？(y/N): ")
        if response.lower() != 'y':
            print("使用现有缓存文件")
            return
    
    # 读取数据
    if not os.path.exists(shot_path):
        print(f"训练数据文件不存在: {shot_path}")
        return
    
    if not os.path.exists(input_path):
        print(f"开发数据文件不存在: {input_path}")
        return
    
    print("正在读取数据文件...")
    train_df = pd.read_csv(shot_path)
    dev_df = pd.read_csv(input_path)
    
    print(f"训练数据: {len(train_df)} 条")
    print(f"开发数据: {len(dev_df)} 条")
    
    # 筛选有效的训练数据
    train_df_valid = train_df[
        (train_df['语言'].notna()) &
        (train_df['文本'].notna()) &
        (train_df['文本'] != "") &
        (train_df['中文'].notna()) &
        (train_df['中文'] != "")
    ].copy()
    
    # 筛选有效的开发数据
    dev_df_valid = dev_df[
        (dev_df.iloc[:, 3].notna()) &  # 语言列
        (dev_df.iloc[:, 4].notna()) &  # 中文列
        (dev_df.iloc[:, 4] != "")
    ].copy()
    
    print(f"有效训练数据: {len(train_df_valid)} 条")
    print(f"有效开发数据: {len(dev_df_valid)} 条")
    
    # 初始化缓存结构
    bleu2_cache = {}
    
    # 按语言分组处理
    languages = dev_df_valid.iloc[:, 3].unique()  # 语言列
    
    total_start_time = time.time()
    
    for lang in languages:
        print(f"\n处理语言: {lang}")
        lang_start_time = time.time()
        
        # 获取该语言的开发数据和训练数据
        dev_lang_data = dev_df_valid[dev_df_valid.iloc[:, 3] == lang]
        train_lang_data = train_df_valid[train_df_valid['语言'] == lang]
        
        if len(train_lang_data) == 0:
            print(f"  警告: 没有找到语言 '{lang}' 的训练数据")
            continue
            
        print(f"  开发数据: {len(dev_lang_data)} 条")
        print(f"  训练数据: {len(train_lang_data)} 条")
        
        # 准备计算任务 - 优化版本，分批处理减少内存占用
        tasks = []
        train_data_list = [(train_idx, train_row['中文']) for train_idx, train_row in train_lang_data.iterrows()]
        
        # 计算合适的批次大小：每个任务处理的train样本数量
        max_train_per_task = min(2000, len(train_data_list))  # 每个任务最多处理2000个train样本
        
        for dev_idx, dev_row in dev_lang_data.iterrows():
            dev_text = dev_row.iloc[4]  # 中文列
            
            # 如果训练数据量大，分批处理
            if len(train_data_list) > max_train_per_task:
                for i in range(0, len(train_data_list), max_train_per_task):
                    train_batch = train_data_list[i:i+max_train_per_task]
                    tasks.append((f"{dev_idx}_{i//max_train_per_task}", dev_text, train_batch, "中文"))
            else:
                # 小数据量，直接处理
                tasks.append((dev_idx, dev_text, train_data_list, "中文"))
        
        print(f"  总计算任务数: {len(tasks)}")
        print(f"  使用 {BLEU2_WORKERS} 个并行工作进程")
        
        # 简化实现：直接尝试使用进程池，失败时使用线程池
        print(f"  尝试使用进程池进行并行计算...")
        
        def test_multiprocessing():
            """测试多进程是否可用"""
            try:
                print(f"    测试进程池，主进程PID: {os.getpid()}")
                
                # 测试1：基本进程创建
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(os.getpid)
                    result = future.result(timeout=10)
                    if result != os.getpid():
                        print(f"    ✓ 进程池基本测试成功，子进程PID: {result}")
                    else:
                        print(f"    ✗ 进程池可能未创建新进程")
                        return False
                
                # 测试2：BLEU2计算函数
                test_args = (99999, "测试文本", [(1, "test"), (2, "测试")], "中文")
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(calculate_bleu2_batch_worker, test_args)
                    result = future.result(timeout=30)
                    print(f"    ✓ BLEU2函数测试成功，结果类型: {type(result)}")
                
                return True
            except Exception as e:
                print(f"    ✗ 进程池测试失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        use_processes = test_multiprocessing()
        
        if use_processes:
            executor_class = ProcessPoolExecutor
            executor_type = "进程池"
        else:
            executor_class = ThreadPoolExecutor
            executor_type = "线程池"
            
        print(f"  使用 {executor_type} 进行并行计算，工作进程数: {BLEU2_WORKERS}")
        
        # 使用选定的执行器进行并行计算
        total_calculations = 0
        print(f"  开始并行计算...")
        print(f"  主进程PID: {os.getpid()}")
        
        with executor_class(max_workers=BLEU2_WORKERS) as executor:
            # 提交所有任务
            print(f"  提交 {len(tasks)} 个计算任务...")
            future_to_task = {}
            
            for i, task in enumerate(tasks):
                if use_processes:
                    future = executor.submit(calculate_bleu2_batch_worker, task)
                else:
                    future = executor.submit(calculate_bleu2_batch_worker, task)  # 线程池也用同样的函数
                future_to_task[future] = (i, task)
                
                # if i % 10 == 0:  # 每10个任务显示一次进度
                #     print(f"    已提交 {i+1}/{len(tasks)} 个任务")
            
            print(f"  所有任务已提交，等待结果...")
            
            # 收集结果
            completed_tasks = 0
            failed_tasks = 0
            with tqdm(total=len(tasks), desc=f"  计算 {lang} BLEU2分数") as pbar:
                for future in future_to_task:
                    try:
                        task_dev_idx, results, calc_count = future.result(timeout=300)  # 5分钟超时
                        
                        # 解析dev_idx（可能包含批次信息）
                        if isinstance(task_dev_idx, str) and '_' in str(task_dev_idx):
                            # 分批任务：dev_idx_batch_num
                            real_dev_idx = int(str(task_dev_idx).split('_')[0])
                        else:
                            # 普通任务
                            real_dev_idx = task_dev_idx
                        
                        if real_dev_idx not in bleu2_cache:
                            bleu2_cache[real_dev_idx] = {}
                        
                        # 合并结果
                        bleu2_cache[real_dev_idx].update(results)
                        completed_tasks += 1
                        total_calculations += calc_count
                        
                        # 每完成20个任务更新一次tqdm描述
                        if completed_tasks % 20 == 0:
                            success_rate = completed_tasks / (completed_tasks + failed_tasks) * 100 if (completed_tasks + failed_tasks) > 0 else 100
                            pbar.set_description(f"  计算 {lang} BLEU2分数 [成功率:{success_rate:.1f}%]")
                            
                    except Exception as e:
                        failed_tasks += 1
                        task_idx, task = future_to_task[future]
                        # print(f"    任务 {task_idx} 失败: {e}")
                        # 对失败的任务，创建空的缓存条目
                        task_dev_idx = task[0]
                        if isinstance(task_dev_idx, str) and '_' in str(task_dev_idx):
                            real_dev_idx = int(str(task_dev_idx).split('_')[0])
                        else:
                            real_dev_idx = task_dev_idx
                        if real_dev_idx not in bleu2_cache:
                            bleu2_cache[real_dev_idx] = {}
                    
                    pbar.update(1)
                
                print(f"    任务完成统计: 成功 {completed_tasks}, 失败 {failed_tasks}, 总计算次数 {total_calculations}")
        
        lang_end_time = time.time()
        print(f"  语言 {lang} 处理完成，耗时: {lang_end_time - lang_start_time:.2f} 秒")
    
    # 保存缓存到文件
    print(f"\n正在保存BLEU2缓存到: {bleu2_cache_file}")
    
    # 转换索引为字符串以便JSON序列化
    cache_for_json = {}
    for dev_idx, train_scores in bleu2_cache.items():
        cache_for_json[str(dev_idx)] = {str(train_idx): score for train_idx, score in train_scores.items()}
    
    with open(bleu2_cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_for_json, f, ensure_ascii=False, indent=2)
    
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"BLEU2缓存计算完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"共计算了 {len(bleu2_cache)} 个开发样本的相似度分数")
    print(f"缓存文件大小: {os.path.getsize(bleu2_cache_file) / 1024 / 1024:.2f} MB")
    print(f"平均每个样本耗时: {total_time / len(bleu2_cache):.3f} 秒")

def precompute_bleu2_cache_simple():
    """简化版本：使用线程池和分词缓存，避免多进程复杂性"""
    print("开始预计算BLEU2相似度分数（简化版本）...")
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 加载分词缓存
    load_tokenize_cache()
    
    # 如果缓存文件已存在，询问是否重新计算
    if os.path.exists(bleu2_cache_file):
        response = input(f"BLEU2缓存文件已存在: {bleu2_cache_file}\n是否重新计算？(y/N): ")
        if response.lower() != 'y':
            print("使用现有缓存文件")
            return
    
    # 读取数据
    if not os.path.exists(shot_path):
        print(f"训练数据文件不存在: {shot_path}")
        return
    
    if not os.path.exists(input_path):
        print(f"开发数据文件不存在: {input_path}")
        return
    
    print("正在读取数据文件...")
    train_df = pd.read_csv(shot_path)
    dev_df = pd.read_csv(input_path)
    
    print(f"训练数据: {len(train_df)} 条")
    print(f"开发数据: {len(dev_df)} 条")
    
    # 筛选有效的训练数据
    train_df_valid = train_df[
        (train_df['语言'].notna()) &
        (train_df['文本'].notna()) &
        (train_df['文本'] != "") &
        (train_df['中文'].notna()) &
        (train_df['中文'] != "")
    ].copy()
    
    # 筛选有效的开发数据
    dev_df_valid = dev_df[
        (dev_df.iloc[:, 3].notna()) &  # 语言列
        (dev_df.iloc[:, 4].notna()) &  # 中文列
        (dev_df.iloc[:, 4] != "")
    ].copy()
    
    print(f"有效训练数据: {len(train_df_valid)} 条")
    print(f"有效开发数据: {len(dev_df_valid)} 条")
    
    # 初始化缓存结构
    bleu2_cache = {}
    
    # 按语言分组处理
    languages = dev_df_valid.iloc[:, 3].unique()  # 语言列
    
    total_start_time = time.time()
    
    for lang in languages:
        print(f"\n处理语言: {lang}")
        lang_start_time = time.time()
        
        # 获取该语言的开发数据和训练数据
        dev_lang_data = dev_df_valid[dev_df_valid.iloc[:, 3] == lang]
        train_lang_data = train_df_valid[train_df_valid['语言'] == lang]
        
        if len(train_lang_data) == 0:
            print(f"  警告: 没有找到语言 '{lang}' 的训练数据")
            continue
            
        print(f"  开发数据: {len(dev_lang_data)} 条")
        print(f"  训练数据: {len(train_lang_data)} 条")
        
        # 为每个开发数据计算与所有同语言训练数据的BLEU2分数
        for dev_idx, dev_row in tqdm(dev_lang_data.iterrows(), 
                                   desc=f"  计算 {lang} BLEU2分数", 
                                   total=len(dev_lang_data)):
            dev_text = dev_row.iloc[4]  # 中文列
            
            if dev_idx not in bleu2_cache:
                bleu2_cache[dev_idx] = {}
            
            # 使用缓存的分词计算与所有同语言训练数据的BLEU2分数
            for train_idx, train_row in train_lang_data.iterrows():
                train_text = train_row['中文']
                bleu2_score = calculate_bleu2_similarity(dev_text, train_text, "中文")
                bleu2_cache[dev_idx][train_idx] = bleu2_score
        
        lang_end_time = time.time()
        print(f"  语言 {lang} 处理完成，耗时: {lang_end_time - lang_start_time:.2f} 秒")
    
    # 保存缓存到文件
    print(f"\n正在保存BLEU2缓存到: {bleu2_cache_file}")
    
    # 转换索引为字符串以便JSON序列化
    cache_for_json = {}
    for dev_idx, train_scores in bleu2_cache.items():
        cache_for_json[str(dev_idx)] = {str(train_idx): score for train_idx, score in train_scores.items()}
    
    with open(bleu2_cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_for_json, f, ensure_ascii=False, indent=2)
    
    # 保存分词缓存
    save_tokenize_cache()
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"BLEU2缓存计算完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"共计算了 {len(bleu2_cache)} 个开发样本的相似度分数")
    print(f"缓存文件大小: {os.path.getsize(bleu2_cache_file) / 1024 / 1024:.2f} MB")
    print(f"平均每个样本耗时: {total_time / len(bleu2_cache):.3f} 秒")

def check_cache_status():
    """检查缓存状态"""
    print("=== 缓存状态检查 ===")
    
    # 检查BLEU2缓存
    if os.path.exists(bleu2_cache_file):
        file_size = os.path.getsize(bleu2_cache_file) / 1024 / 1024
        print(f"✓ BLEU2缓存文件存在: {bleu2_cache_file} ({file_size:.2f} MB)")
        
        try:
            with open(bleu2_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            print(f"  - 包含 {len(cache_data)} 个开发样本的分数")
        except Exception as e:
            print(f"  - 加载失败: {e}")
    else:
        print(f"✗ BLEU2缓存文件不存在: {bleu2_cache_file}")
    
    # 检查相似度排序缓存
    if os.path.exists(similarity_ranking_cache_file):
        file_size = os.path.getsize(similarity_ranking_cache_file) / 1024 / 1024
        print(f"✓ 相似度排序缓存文件存在: {similarity_ranking_cache_file} ({file_size:.2f} MB)")
        
        try:
            with open(similarity_ranking_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            print(f"  - 包含 {len(cache_data)} 个开发样本的排序")
            
            # 检查一个示例
            if cache_data:
                sample_key = list(cache_data.keys())[0]
                sample_langs = list(cache_data[sample_key].keys())
                print(f"  - 示例：dev_idx={sample_key}, 语言={sample_langs}")
        except Exception as e:
            print(f"  - 加载失败: {e}")
    else:
        print(f"✗ 相似度排序缓存文件不存在: {similarity_ranking_cache_file}")
        print("  请运行: python script.py --generate-similarity-ranking")
    
    # 检查训练数据
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

def generate_similarity_ranking_cache():
    """基于现有的BLEU2缓存生成相似度排序缓存"""
    print("开始生成相似度排序缓存...")
    
    # 加载现有的BLEU2缓存
    bleu2_cache = load_bleu2_cache()
    if bleu2_cache is None:
        print("请先运行预计算生成BLEU2缓存")
        return
    
    # 读取数据
    if not os.path.exists(shot_path):
        print(f"训练数据文件不存在: {shot_path}")
        return
    
    if not os.path.exists(input_path):
        print(f"开发数据文件不存在: {input_path}")
        return
    
    print("正在读取数据文件...")
    train_df = pd.read_csv(shot_path)
    dev_df = pd.read_csv(input_path)
    
    # 筛选有效的训练数据
    train_df_valid = train_df[
        (train_df['语言'].notna()) &
        (train_df['文本'].notna()) &
        (train_df['文本'] != "") &
        (train_df['中文'].notna()) &
        (train_df['中文'] != "")
    ].copy()
    
    # 筛选有效的开发数据
    dev_df_valid = dev_df[
        (dev_df.iloc[:, 3].notna()) &  # 语言列
        (dev_df.iloc[:, 4].notna()) &  # 中文列
        (dev_df.iloc[:, 4] != "")
    ].copy()
    
    # 按语言分组处理
    languages = dev_df_valid.iloc[:, 3].unique()  # 语言列
    
    print("正在生成相似度排序缓存...")
    similarity_ranking_cache = {}
    
    for dev_idx, train_scores in tqdm(bleu2_cache.items(), desc="生成排序缓存"):
        similarity_ranking_cache[dev_idx] = {}
        
        # 按语言分组并排序
        for lang in languages:
            # 获取该语言的训练数据
            train_lang_data = train_df_valid[train_df_valid['语言'] == lang]
            
            # 收集该语言下的分数
            lang_scores = []
            for train_idx in train_lang_data.index:
                if train_idx in train_scores:
                    bleu_score = train_scores[train_idx]
                    lang_scores.append((bleu_score, train_idx))
            
            # 按BLEU2分数降序排序
            lang_scores.sort(key=lambda x: x[0], reverse=True)
            # 只保存train_idx列表
            similarity_ranking_cache[dev_idx][lang] = [train_idx for _, train_idx in lang_scores]
    
    # 保存相似度排序缓存
    save_similarity_ranking_cache(similarity_ranking_cache)
    
    print(f"相似度排序缓存生成完成！")
    print(f"共生成 {len(similarity_ranking_cache)} 个开发样本的排序")
    print(f"缓存文件大小: {os.path.getsize(similarity_ranking_cache_file) / 1024 / 1024:.2f} MB")

def precompute_bleu2_cache():
    """原版本：为了向后兼容，调用简化版本"""
    precompute_bleu2_cache_simple()

def load_bleu2_cache():
    """加载BLEU2缓存"""
    if not os.path.exists(bleu2_cache_file):
        print(f"BLEU2缓存文件不存在: {bleu2_cache_file}")
        print("请先运行预计算: python script.py --precompute-bleu2")
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

class DeepSeekTranslator:
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
        
    def load_examples(self, target_lang, current_text=None, dev_idx=None, current_show=None):
        """从训练数据加载指定语言的翻译示例，基于缓存的相似度排序选择最相似的示例，只在同一部剧的语料中搜索"""
        # 为缓存添加dev_idx和show信息来区分不同的查询
        cache_key = f"{target_lang}_{current_show}_{dev_idx if dev_idx is not None else hash(current_text) if current_text else 'default'}"
        
        if cache_key in self.examples_cache:
            return self.examples_cache[cache_key]
            
        examples = []
        if self.n_shot > 0 and self.train_data is not None:
            try:
                # 筛选出目标语言的数据，且有有效翻译结果的，并且属于同一部剧
                base_conditions = [
                    (self.train_data['语言'] == target_lang),  # 语言列匹配
                    (self.train_data['文本'].notna()),  # 有翻译结果
                    (self.train_data['文本'] != ""),  # 翻译结果非空
                    (self.train_data['中文'].notna()),  # 中文文本非空
                    (self.train_data['中文'] != "")
                ]
                
                # 如果提供了剧集信息，则只在同一部剧中搜索
                if current_show:
                    # 从剧集名称中提取剧集标识（如"电视剧1"）
                    show_id = current_show.split(' ')[0] if ' ' in current_show else current_show
                    base_conditions.append(self.train_data['片段名'].str.contains(show_id, na=False))
                
                # 应用所有筛选条件
                lang_data = self.train_data[
                    base_conditions[0] & base_conditions[1] & base_conditions[2] & 
                    base_conditions[3] & base_conditions[4] & 
                    (base_conditions[5] if len(base_conditions) > 5 else True)
                ]
                
                # 计算总的语言数据（用于保底）
                total_lang_data = self.train_data[
                    base_conditions[0] & base_conditions[1] & base_conditions[2] & 
                    base_conditions[3] & base_conditions[4]
                ]
                
                # 首先尝试从同一部剧中获取示例
                selected_indices = set()  # 用于追踪已选择的索引，避免重复
                examples = []
                
                # 阶段1：从同一部剧中选择示例
                same_show_examples = 0
                if len(lang_data) > 0:
                    if dev_idx is not None and self.similarity_ranking_cache and dev_idx in self.similarity_ranking_cache:
                        # 优先使用相似度排序缓存（最快）
                        if target_lang in self.similarity_ranking_cache[dev_idx]:
                            sorted_train_indices = self.similarity_ranking_cache[dev_idx][target_lang]
                            # 筛选出在当前lang_data中存在的索引
                            lang_data_indices = set(lang_data.index)
                            valid_indices = [idx for idx in sorted_train_indices if idx in lang_data_indices]
                            same_show_count = min(len(valid_indices), self.n_shot)
                            
                            for train_idx in valid_indices[:same_show_count]:
                                row = self.train_data.loc[train_idx]
                                chinese_text = row['中文']
                                translated_text = row['文本']
                                examples.append((chinese_text, translated_text))
                                selected_indices.add(train_idx)
                                same_show_examples += 1
                    elif dev_idx is not None and self.bleu2_cache and dev_idx in self.bleu2_cache:
                        # 备选方案：使用BLEU2缓存并重新排序
                        cached_scores = self.bleu2_cache[dev_idx]
                        lang_scores = []
                        for train_idx in lang_data.index:
                            if train_idx in cached_scores:
                                bleu_score = cached_scores[train_idx]
                                lang_scores.append((bleu_score, train_idx))
                        
                        lang_scores.sort(key=lambda x: x[0], reverse=True)
                        same_show_count = min(len(lang_scores), self.n_shot)
                        
                        for bleu_score, train_idx in lang_scores[:same_show_count]:
                            row = self.train_data.loc[train_idx]
                            chinese_text = row['中文']
                            translated_text = row['文本']
                            examples.append((chinese_text, translated_text))
                            selected_indices.add(train_idx)
                            same_show_examples += 1
                    elif current_text:
                        # 实时计算BLEU2分数（备用方案）
                        bleu_scores = []
                        for _, row in lang_data.iterrows():
                            chinese_text = row['中文']
                            bleu_score = calculate_bleu2_similarity(current_text, chinese_text, "中文")
                            bleu_scores.append((bleu_score, row))
                        
                        bleu_scores.sort(key=lambda x: x[0], reverse=True)
                        same_show_count = min(len(bleu_scores), self.n_shot)
                        
                        for bleu_score, row in bleu_scores[:same_show_count]:
                            chinese_text = row['中文']
                            translated_text = row['文本']
                            examples.append((chinese_text, translated_text))
                            selected_indices.add(row.name)  # row.name 是索引
                            same_show_examples += 1
                    else:
                        # 如果没有current_text，随机选择
                        available_count = min(len(lang_data), self.n_shot)
                        sampled_data = lang_data.sample(n=available_count, random_state=42)
                        for _, row in sampled_data.iterrows():
                            chinese_text = row['中文']
                            translated_text = row['文本']
                            examples.append((chinese_text, translated_text))
                            selected_indices.add(row.name)
                            same_show_examples += 1

                # 阶段2：如果同一部剧的示例不足，从其他剧集补充
                remaining_needed = self.n_shot - same_show_examples
                if remaining_needed > 0:
                    # 获取所有语言数据（不限剧集）
                    all_lang_data = total_lang_data  # 之前计算的不限剧集的数据
                    
                    if dev_idx is not None and self.similarity_ranking_cache and dev_idx in self.similarity_ranking_cache:
                        if target_lang in self.similarity_ranking_cache[dev_idx]:
                            sorted_train_indices = self.similarity_ranking_cache[dev_idx][target_lang]
                            # 筛选出在所有语言数据中存在且未被选择的索引
                            all_lang_data_indices = set(all_lang_data.index)
                            valid_indices = [idx for idx in sorted_train_indices 
                                           if idx in all_lang_data_indices and idx not in selected_indices]
                            
                            for train_idx in valid_indices[:remaining_needed]:
                                row = self.train_data.loc[train_idx]
                                chinese_text = row['中文']
                                translated_text = row['文本']
                                examples.append((chinese_text, translated_text))
                                selected_indices.add(train_idx)
                    elif dev_idx is not None and self.bleu2_cache and dev_idx in self.bleu2_cache:
                        cached_scores = self.bleu2_cache[dev_idx]
                        lang_scores = []
                        for train_idx in all_lang_data.index:
                            if train_idx in cached_scores and train_idx not in selected_indices:
                                bleu_score = cached_scores[train_idx]
                                lang_scores.append((bleu_score, train_idx))
                        
                        lang_scores.sort(key=lambda x: x[0], reverse=True)
                        
                        for bleu_score, train_idx in lang_scores[:remaining_needed]:
                            row = self.train_data.loc[train_idx]
                            chinese_text = row['中文']
                            translated_text = row['文本']
                            examples.append((chinese_text, translated_text))
                            selected_indices.add(train_idx)
                    elif current_text:
                        # 实时计算剩余数据的BLEU2分数
                        remaining_data = all_lang_data[~all_lang_data.index.isin(selected_indices)]
                        bleu_scores = []
                        for _, row in remaining_data.iterrows():
                            chinese_text = row['中文']
                            bleu_score = calculate_bleu2_similarity(current_text, chinese_text, "中文")
                            bleu_scores.append((bleu_score, row))
                        
                        bleu_scores.sort(key=lambda x: x[0], reverse=True)
                        
                        for bleu_score, row in bleu_scores[:remaining_needed]:
                            chinese_text = row['中文']
                            translated_text = row['文本']
                            examples.append((chinese_text, translated_text))
                            selected_indices.add(row.name)
                    else:
                        # 随机从剩余数据中选择
                        remaining_data = all_lang_data[~all_lang_data.index.isin(selected_indices)]
                        if len(remaining_data) > 0:
                            sample_count = min(len(remaining_data), remaining_needed)
                            sampled_data = remaining_data.sample(n=sample_count, random_state=42)
                            for _, row in sampled_data.iterrows():
                                chinese_text = row['中文']
                                translated_text = row['文本']
                                examples.append((chinese_text, translated_text))
                                selected_indices.add(row.name)

            except Exception as e:
                print(f"加载示例时出错: {e}")
                examples = []
        
        self.examples_cache[cache_key] = examples
        return examples
        
    def create_system_prompt(self, source_lang, target_lang, current_text=None, dev_idx=None, current_show=None):
        """创建系统提示词，包含 few-shot 示例"""
        if self.n_shot == 0:
            # 原有的系统提示词
            system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。用户将输入中文文本，请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
        else:
            # 包含示例的系统提示词，基于缓存的BLEU2分数选择，只在同一部剧中搜索
            examples = self.load_examples(target_lang, current_text, dev_idx, current_show)
            
            if examples:
                # 构建示例说明，区分同剧集和跨剧集示例
                if current_show:
                    examples_text = f"\n\n以下是一些翻译示例供参考（按相似度排序，优先使用同剧集：{current_show}）：\n"
                else:
                    examples_text = f"\n\n以下是一些翻译示例供参考（按相似度排序）：\n"
                
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
    
    def translate_text(self, text, source_lang, target_lang, dev_idx=None, current_show=None):
        """使用DeepSeek API翻译文本，带重试和指数退避"""
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 创建系统和用户提示，传入dev_idx和剧集信息用于示例选择
                system_prompt = self.create_system_prompt(source_lang, target_lang, text, dev_idx, current_show)
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
        index, (text, source_lang, target_lang, dev_idx, show_name) = index_and_data
        if pd.isna(text) or text == "":
            return index, ""
        
        # 添加单个任务的调试信息
        try:
            result = translator.translate_text(text, source_lang, target_lang, dev_idx, show_name)
            return index, result
        except Exception as e:
            print(f"任务 {index} (dev_idx={dev_idx}, show={show_name}) 翻译失败: {e}")
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
    
    # 准备翻译数据（包含dev_idx和剧集信息）
    texts_and_langs_with_idx = []
    indices = []
    
    for idx, row in need_translation.iterrows():
        chinese_text = row.iloc[4]  # 中文列
        target_lang_chinese = row.iloc[3]  # 语言列
        show_name = row.iloc[1] if len(row) > 1 else None  # 片段名列
        target_lang_english = lang_map.get(target_lang_chinese, target_lang_chinese)
        
        texts_and_langs_with_idx.append((chinese_text, "中文", target_lang_chinese, idx, show_name))  # 添加dev_idx和剧集信息
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

def sync_translate_one(text, source_lang, target_lang, n_shot=0, show_name=None):
    """同步翻译单个文本（用于测试）"""
    translator = DeepSeekTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    return translator.translate_text(text, source_lang, target_lang, current_show=show_name)

def test_translation(n_shot=0):
    """测试翻译功能"""
    print("测试DeepSeek翻译...")
    if n_shot > 0:
        print(f"使用 {n_shot}-shot 学习模式进行测试")
    
    # 检查API密钥
    if API_KEY == "<DeepSeek API Key>":
        print("请先设置DeepSeek API密钥！")
        return
    
    test_cases = [
        ("你好，世界！", "中文", "英语"),
        ("这是一个测试句子。", "中文", "马来语"),
        ("今天天气很好。", "中文", "泰语")
    ]
    
    # 创建翻译器以便获取完整输入
    translator = DeepSeekTranslator(API_KEY, BASE_URL, MODEL_NAME, n_shot=n_shot)
    
    for i, (text, source_lang, target_lang) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试案例 {i}")
        print(f"{'='*80}")
        print(f"原文 ({source_lang}): {text}")
        print(f"目标语言: {target_lang}")
        
        # 获取完整的系统提示词（测试时可以指定剧集）
        test_show = "电视剧1 EP09"  # 测试用的剧集名称
        system_prompt = translator.create_system_prompt(source_lang, target_lang, text, current_show=test_show)
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
        result = translator.translate_text(text, source_lang, target_lang, current_show=test_show)
        print(f"译文 ({target_lang}): {result}")
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用DeepSeek API进行翻译")
    parser.add_argument("--test", action="store_true", help="运行翻译测试")
    parser.add_argument("--api-key", type=str, help="DeepSeek API密钥")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, 
                       help="最大并发请求数")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, 
                       help="请求间隔（秒）")
    parser.add_argument("--n-shot", type=int, default=30,
                       help=f"Few-shot 示例数量 (0表示不使用few-shot，默认: {DEFAULT_N_SHOT})")
    parser.add_argument("--precompute-bleu2", action="store_true",
                       help="预计算BLEU2相似度分数并缓存到文件")
    parser.add_argument("--precompute-bleu2-fast", action="store_true",
                       help="使用优化版本预计算BLEU2相似度分数（多进程，更快但可能不稳定）")
    parser.add_argument("--precompute-bleu2-simple", action="store_true",
                       help="使用简化版本预计算BLEU2相似度分数（单进程+缓存，稳定）")
    parser.add_argument("--generate-similarity-ranking", action="store_true",
                       help="基于现有BLEU2缓存生成相似度排序缓存（提高推理速度）")
    parser.add_argument("--check-cache", action="store_true",
                       help="检查缓存状态")
    parser.add_argument("--low-concurrency", action="store_true",
                       help="使用低并发模式（推荐在网络不稳定时使用）")
    parser.add_argument("--bleu2-workers", type=int, default=BLEU2_WORKERS,
                       help=f"BLEU2计算的并行工作进程数（默认: {BLEU2_WORKERS}）")
    
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
    BLEU2_WORKERS = args.bleu2_workers
    
    # 设置 few-shot 参数
    n_shot = args.n_shot
    
    if args.precompute_bleu2_fast:
        precompute_bleu2_cache_optimized()
    elif args.precompute_bleu2_simple:
        precompute_bleu2_cache_simple()
    elif args.precompute_bleu2:
        precompute_bleu2_cache()
    elif args.generate_similarity_ranking:
        generate_similarity_ranking_cache()
    elif args.check_cache:
        check_cache_status()
    elif args.test:
        test_translation(n_shot=n_shot)
    else:
        translate_deepseek(n_shot=n_shot)