import os
# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# 如果需要，也可以设置no_proxy
# os.environ['no_proxy'] = 'localhost,127.0.0.1'

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import json
import requests

# 过滤警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 路径配置
base_dir = os.getcwd()
# input_path = os.path.join(base_dir, "output/testa_asr.csv")  # 使用ASR结果作为输入
# gemini_output_path = os.path.join(base_dir, "output/gemini/result2.csv")
input_path = os.path.join(base_dir, "data/text_data/dev.csv")  # 使用ASR结果作为输入
gemini_output_path = os.path.join(base_dir, "output/gemini/dev_g.csv")

# 确保输出目录存在
os.makedirs(os.path.dirname(gemini_output_path), exist_ok=True)

# 语言映射
lang_map = {
    "马来语": "Malay",
    "泰语": "Thai", 
    "英语": "English"
}

# Gemini API配置
API_KEYS = [
    "AIzaSyAfOE7TG4C6mYF-en8DVUMKrU-GgAOaAX0",
    "AIzaSyDzEM-8qtZLYSjxK9OxUiXvjm8v96nKX3g", 
    "AIzaSyAUyX_M64ZsLM7ds2hTEMCTF_CE4ySLPZg",
    "AIzaSyCIg0DuyUkrfPWGbhwJ5Jq-SMMAj9WbO20",
    "AIzaSyBHbYVAi1ewtp88teVnu_zLXoJU1ORk3gE"
]  # 多个Gemini API Key
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# 并发配置  
MAX_CONCURRENT_REQUESTS = len(API_KEYS)  # 总并发数等于key的数量
REQUEST_DELAY = 0.5  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_FACTOR = 2  # 退避因子

class GeminiTranslator:
    def __init__(self, api_keys, model_name, base_url):
        # 为每个API key创建一个客户端
        self.clients = []
        self.api_keys = api_keys
        for api_key in api_keys:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.clients.append(client)
        self.model_name = model_name
        self.current_client_index = 0
        
    def get_next_client(self):
        """轮询获取下一个可用的客户端"""
        client = self.clients[self.current_client_index]
        api_key = self.api_keys[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client, api_key
        
    def create_system_prompt(self, source_lang, target_lang):
        """创建系统提示词"""
        system_prompt = f"""你是一个专业的多语言翻译专家，擅长将{source_lang}翻译成{target_lang}。用户将输入中文文本，请只返回翻译后的{target_lang}文本，不要包含任何解释或额外信息。"""
        return system_prompt
    
    def translate_text(self, text, source_lang, target_lang, client_index=None):
        """使用Gemini API翻译文本，带重试和指数退避"""
        # 如果指定了客户端索引，使用指定的客户端，否则轮询获取
        if client_index is not None:
            client = self.clients[client_index]
            api_key = self.api_keys[client_index]
        else:
            client, api_key = self.get_next_client()
            
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 创建系统和用户提示
                system_prompt = self.create_system_prompt(source_lang, target_lang)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    reasoning_effort="none",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    # temperature=1.3,
                    max_tokens=128,
                    stream=False
                )
                
                translated_text = response.choices[0].message.content.strip()
                time.sleep(REQUEST_DELAY)  # 添加请求延迟
                return translated_text
                
            except Exception as e:
                print(f"翻译过程中出错 (API Key: {api_key[:20]}...): {str(e)}")
                if attempt < MAX_RETRIES:
                    wait_time = REQUEST_DELAY * (BACKOFF_FACTOR ** attempt)
                    print(f"等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"达到最大重试次数，跳过当前文本")
                    return ""
        
        return ""

def translate_batch_parallel(translator, texts_and_langs, max_workers, pbar=None):
    """使用线程池并行翻译文本，每个线程使用不同的API key"""
    results = [""] * len(texts_and_langs)
    
    def translate_single(index_and_data):
        index, (text, source_lang, target_lang) = index_and_data
        if pd.isna(text) or text == "":
            return index, ""
        # 根据索引分配客户端，确保每个API key的并发为1
        client_index = index % len(translator.clients)
        result = translator.translate_text(text, source_lang, target_lang, client_index)
        return index, result
    
    # 使用与API key数量相等的线程数，确保每个key的并发为1
    actual_max_workers = min(max_workers, len(translator.clients))
    
    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        indexed_data = [(i, data) for i, data in enumerate(texts_and_langs)]
        futures = [executor.submit(translate_single, item) for item in indexed_data]
        
        for future in futures:
            try:
                index, result = future.result()
                results[index] = result
                if pbar:
                    pbar.update(1)
            except Exception as e:
                print(f"翻译任务出错: {e}")
                if pbar:
                    pbar.update(1)
    
    return results

def translate_gemini():
    """主翻译函数"""
    print("开始使用Gemini API进行翻译...")
    
    # 检查API密钥
    if len(API_KEYS) == 0 or API_KEYS[0] == "<Gemini API Key>":
        print("警告：请在脚本中设置您的Gemini API密钥！")
        print("请将API_KEYS变量替换为您的实际API密钥")
        return
    
    # 读取ASR结果文件
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        print("请先运行ASR处理生成输入文件")
        return
    
    df = pd.read_csv(input_path)
    print(f"读取到 {len(df)} 条数据")
    
    # 检查是否有已存在的翻译结果
    if 'answer' not in df.columns:
        df['answer'] = ""
    
    # 获取需要翻译的数据
    need_translation = df[df['answer'].isna() | (df['answer'] == "")]
    
    if len(need_translation) == 0:
        print("所有数据已翻译完成！")
        return
        
    print(f"需要翻译 {len(need_translation)} 条数据")
    
    # 准备翻译数据
    texts_and_langs = []
    indices = []
    
    for idx, row in need_translation.iterrows():
        chinese_text = row.iloc[4]  # 中文列
        target_lang_chinese = row.iloc[3]  # 语言列
        target_lang_english = lang_map.get(target_lang_chinese, target_lang_chinese)
        
        texts_and_langs.append((chinese_text, "中文", target_lang_chinese))
        indices.append(idx)
    
    # 创建Gemini翻译器
    translator = GeminiTranslator(API_KEYS, MODEL_NAME, BASE_URL)
    
    # 分批翻译，每10条保存一次
    batch_size = MAX_CONCURRENT_REQUESTS  # 每批处理10条
    total_batches = (len(texts_and_langs) + batch_size - 1) // batch_size
    
    print(f"开始翻译，共 {total_batches} 批，每批 {batch_size} 条...")
    print(f"使用 {len(API_KEYS)} 个API Key，总并发线程数: {MAX_CONCURRENT_REQUESTS}")
    print(f"API Keys: {[key[:20] + '...' for key in API_KEYS]}")
    
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    
    # 创建总进度条
    with tqdm(total=len(texts_and_langs), desc="总翻译进度") as pbar:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(texts_and_langs))
            batch_texts_and_langs = texts_and_langs[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]
            
            # 翻译当前批次
            batch_start_time = time.time()
            batch_results = translate_batch_parallel(translator, batch_texts_and_langs, MAX_CONCURRENT_REQUESTS, pbar)
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
    print(f"总翻译条数: {len(texts_and_langs)} 条")
    print(f"成功翻译: {total_successful} 条")
    print(f"翻译失败: {total_failed} 条")
    print(f"成功率: {(total_successful/len(texts_and_langs)*100):.1f}%")
    print(f"平均每条翻译耗时: {elapsed_time/len(texts_and_langs):.2f} 秒")
    print(f"最终结果已保存到: {gemini_output_path}")

def sync_translate_one(text, source_lang, target_lang):
    """同步翻译单个文本（用于测试）"""
    translator = GeminiTranslator(API_KEYS, MODEL_NAME, BASE_URL)
    return translator.translate_text(text, source_lang, target_lang)

def test_translation():
    """测试翻译功能"""
    print("测试Gemini翻译...")
    
    # 检查API密钥
    if len(API_KEYS) == 0 or API_KEYS[0] == "<Gemini API Key>":
        print("请先设置Gemini API密钥！")
        return
    
    test_cases = [
        ("你好，世界！", "中文", "英语"),
        ("这是一个测试句子。", "中文", "马来语"),
        ("今天天气很好。", "中文", "泰语")
    ]
    
    for text, source_lang, target_lang in test_cases:
        print(f"\n原文 ({source_lang}): {text}")
        result = sync_translate_one(text, source_lang, target_lang)
        print(f"译文 ({target_lang}): {result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用Gemini API进行翻译")
    parser.add_argument("--test", action="store_true", help="运行翻译测试")
    parser.add_argument("--api-key", type=str, help="单个Gemini API密钥（会覆盖默认的多个密钥）")
    parser.add_argument("--api-keys", type=str, nargs='+', help="多个Gemini API密钥")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, 
                       help="最大并发请求数")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, 
                       help="请求间隔（秒）")
    
    args = parser.parse_args()
    
    # 如果提供了API密钥，则更新全局变量
    if args.api_keys:
        API_KEYS = args.api_keys
        MAX_CONCURRENT_REQUESTS = len(API_KEYS)
    elif args.api_key:
        API_KEYS = [args.api_key]
        MAX_CONCURRENT_REQUESTS = 1
    
    # 更新配置
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    REQUEST_DELAY = args.delay
    
    if args.test:
        test_translation()
    else:
        translate_gemini()