import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 禁用torch编译优化，避免FX符号追踪冲突
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"
import pandas as pd
from tqdm import tqdm
import warnings
import time
import torch
torch.set_float32_matmul_precision('high')
# 禁用torch.compile以避免FX追踪冲突
torch._dynamo.config.disable = True
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 过滤警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 语言映射
lang_map = {
    "马来语": "Malay",
    "泰语": "Thai", 
    "英语": "English"
}

# GemmaX2 模型配置
MODEL_ID = "/mnt/gold/lz/trans/models/ft/GemmaX2-28-2B-v0.1-ft-full-5/checkpoint-4950"
# 由于设置了CUDA_VISIBLE_DEVICES="7"，GPU 7会被映射为cuda:0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128

# 推理配置
BATCH_SIZE = 64  # 批处理大小
REQUEST_DELAY = 0.1  # 请求间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
BACKOFF_FACTOR = 2  # 退避因子

# 路径配置
base_dir = os.path.dirname(os.getcwd())
# input_path = os.path.join(base_dir, "data/text_data/dev.csv")
# gemma_output_path = os.path.join(base_dir, f"output/{MODEL_ID.split('/')[-2]}/dev_translate_{MODEL_ID.split('/')[-1].split('-')[-1]}.csv")
input_path = os.path.join(base_dir, "output/testa_asr.csv")
gemma_output_path = os.path.join(base_dir, f"output/{MODEL_ID.split('/')[-2]}/result.csv")

# 确保输出目录存在
os.makedirs(os.path.dirname(gemma_output_path), exist_ok=True)

class GemmaX2Translator:
    def __init__(self, model_id, device):
        """初始化GemmaX2翻译器"""
        print(f"正在加载模型 {model_id}...")
        self.device = device
        
        # 如果是CUDA设备，确保设备存在并设置为当前设备
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                device_id = int(device.split(":")[1]) if ":" in device else 0
                if device_id >= torch.cuda.device_count():
                    print(f"警告：设备 {device} 不存在，使用 cuda:0")
                    self.device = "cuda:0"
                torch.cuda.set_device(self.device)
            else:
                print("警告：CUDA不可用，切换到CPU")
                self.device = "cpu"
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/gold/lz/trans/models/GemmaX2-28-2B-v0.1",
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 设置pad_token如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 显式将模型移动到指定设备，避免多GPU设备不匹配问题
        print(f"将模型移动到设备: {self.device}")
        self.model = self.model.to(self.device)
            
        self.model.eval()
        
        # 清理GPU缓存
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        print(f"模型已加载到 {self.device}")
        
    def create_translation_prompt(self, text, source_lang, target_lang):
        """创建翻译提示词"""
        prompt = f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {text}\n{target_lang}:"
        return prompt
    
    def translate_text(self, text, source_lang, target_lang):
        """使用GemmaX2模型翻译文本，带重试和指数退避"""
        if pd.isna(text) or text == "":
            return ""
            
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 创建翻译提示
                prompt = self.create_translation_prompt(text, source_lang, target_lang)
                
                # 对输入进行tokenize
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=64
                ).to(self.device)
                
                # 生成翻译
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS
                    )
                
                # 解码输出
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                translated_text = full_response.split(":")[-1].strip()

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
    
    def translate_batch(self, texts_and_langs):
        """批量翻译文本"""
        if not texts_and_langs:
            return []
            
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 创建所有提示
                prompts = []
                for text, source_lang, target_lang in texts_and_langs:
                    if pd.isna(text) or text == "":
                        prompts.append("")
                    else:
                        prompt = self.create_translation_prompt(text, source_lang, target_lang)
                        prompts.append(prompt)
                
                # 批量tokenize
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                # 批量生成翻译
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS
                    )
                
                results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                results = [result.split(f":")[-1].replace("\n", " ").strip() for result in results]

                return results
                
            except Exception as e:
                print(f"批量翻译过程中出错: {str(e)}")
                if attempt < MAX_RETRIES:
                    wait_time = REQUEST_DELAY * (BACKOFF_FACTOR ** attempt)
                    print(f"等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{MAX_RETRIES + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"达到最大重试次数，返回空结果")
                    return [""] * len(texts_and_langs)
        
        return [""] * len(texts_and_langs)

def translate_batch_batch(translator, texts_and_langs):
    """批量翻译文本"""
    results = translator.translate_batch(texts_and_langs)
    return results

def translate_gemma():
    """主翻译函数"""
    print("开始使用GemmaX2模型进行翻译...")
    
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
        
        texts_and_langs.append((chinese_text, "Chinese", target_lang_english))
        indices.append(idx)
    
    # 创建GemmaX2翻译器
    translator = GemmaX2Translator(MODEL_ID, DEVICE)
    
    # 分批处理
    total_batches = (len(texts_and_langs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"开始批量翻译，共 {total_batches} 批，每批 {BATCH_SIZE} 条...")
    print(f"设备: {DEVICE}")
    
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    
    # 分批串行翻译
    for batch_idx in tqdm(range(total_batches)):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(texts_and_langs))
        batch_texts_and_langs = texts_and_langs[batch_start:batch_end]
        batch_indices = indices[batch_start:batch_end]
        
        # 翻译当前批次
        batch_start_time = time.time()
        batch_results = translate_batch_batch(translator, batch_texts_and_langs)
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
        df.to_csv(gemma_output_path, index=False)
        
        # 更新总计数
        total_successful += batch_successful
        total_failed += batch_failed
        
        # 显示批次统计
        batch_time = batch_end_time - batch_start_time
        total_processed = total_successful + total_failed
        success_rate = (total_successful / total_processed * 100) if total_processed > 0 else 0
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== 翻译完成 ===")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"总翻译条数: {len(texts_and_langs)} 条")
    print(f"成功翻译: {total_successful} 条")
    print(f"翻译失败: {total_failed} 条")
    print(f"成功率: {(total_successful/len(texts_and_langs)*100):.1f}%")
    print(f"平均每条翻译耗时: {elapsed_time/len(texts_and_langs):.2f} 秒")
    print(f"最终结果已保存到: {gemma_output_path}")

def sync_translate_one(text, source_lang, target_lang):
    """同步翻译单个文本（用于测试）"""
    translator = GemmaX2Translator(MODEL_ID, DEVICE)
    return translator.translate_text(text, source_lang, target_lang)

def test_translation():
    """测试翻译功能"""
    print("测试GemmaX2翻译...")
    
    test_cases = [
        ("你好，世界！", "Chinese", "English"),
        ("这是一个测试句子。", "Chinese", "Malay"),
        ("今天天气很好。", "Chinese", "Thai")
    ]
    
    translator = GemmaX2Translator(MODEL_ID, DEVICE)
    
    for text, source_lang, target_lang in test_cases:
        print(f"\n原文 ({source_lang}): {text}")
        result = translator.translate_text(text, source_lang, target_lang)
        print(f"译文 ({target_lang}): {result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用GemmaX2模型进行翻译")
    parser.add_argument("--test", action="store_true", help="运行翻译测试")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="模型ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="批处理大小")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, 
                       help="请求间隔（秒）")
    parser.add_argument("--device", type=str, default=DEVICE, 
                       help="设备选择 (cuda/cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                       help="最大生成token数")
    
    args = parser.parse_args()
    
    # 更新配置
    MODEL_ID = args.model_id
    BATCH_SIZE = args.batch_size
    REQUEST_DELAY = args.delay
    DEVICE = args.device
    MAX_NEW_TOKENS = args.max_new_tokens
    
    if args.test:
        test_translation()
    else:
        translate_gemma()
