import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pandas as pd
from tqdm import tqdm
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 过滤警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 路径配置
base_dir = os.getcwd()
model_path = os.path.join(base_dir, "models/opus-mt-zh-ms")
input_path = os.path.join(base_dir, "data/text_data/dev.csv")  
output_path = os.path.join(base_dir, "output/opus/dev_ms.csv")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 全局变量存储模型
opus_model = None
opus_tokenizer = None

def init_opus_model():
    """初始化OPUS翻译模型"""
    global opus_model, opus_tokenizer
    if opus_model is None:
        print("正在加载OPUS中文-马来语翻译模型...")
        try:
            opus_tokenizer = AutoTokenizer.from_pretrained(model_path)
            opus_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            # 将模型移到指定的GPU
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            opus_model = opus_model.to(device)
            print(f"OPUS模型加载成功，设备: {device}")
        except Exception as e:
            print(f"OPUS模型加载失败: {e}")
            return None, None
    return opus_model, opus_tokenizer

def translate_text_opus(text):
    """使用OPUS模型翻译中文文本到马来语"""
    global opus_model, opus_tokenizer
    
    if opus_model is None or opus_tokenizer is None:
        print("模型未初始化")
        return ""
    
    try:
        # 获取模型所在的设备
        device = next(opus_model.parameters()).device
        
        # 编码输入文本
        input_ids = opus_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids
        
        # 将输入移到与模型相同的设备
        input_ids = input_ids.to(device)
        
        # 生成翻译
        with torch.no_grad():
            outputs = opus_model.generate(
                input_ids=input_ids,
                num_beams=5,
                max_length=512,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # 解码输出
        translated_text = opus_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text.strip()
        
    except Exception as e:
        print(f"翻译过程中出错: {e}")
        return ""

def translate_malay_data():
    """翻译马来语数据"""
    print("开始使用OPUS模型翻译马来语数据...")
    
    # 初始化模型
    model, tokenizer = init_opus_model()
    if model is None:
        print("模型初始化失败，退出程序")
        return
    
    # 读取数据文件
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"读取到 {len(df)} 条数据")
    
    # 过滤出马来语数据
    malay_data = df[df['语言'] == '马来语'].copy()
    print(f"找到 {len(malay_data)} 条马来语数据")
    
    if len(malay_data) == 0:
        print("没有找到马来语数据")
        return
    
    # 添加翻译结果列
    malay_data['answer'] = ""
    
    # 翻译处理
    print("开始翻译处理...")
    for idx, row in tqdm(malay_data.iterrows(), total=len(malay_data), desc="翻译进度"):
        chinese_text = row['中文']
        if pd.isna(chinese_text) or chinese_text == "":
            continue
            
        # 使用OPUS模型翻译
        translated_text = translate_text_opus(chinese_text)
        malay_data.loc[idx, 'answer'] = translated_text
        
        # 可选：显示翻译示例
        if idx == malay_data.index[0]:  # 显示第一个翻译示例
            print(f"\n翻译示例:")
            print(f"原文(中文): {chinese_text}")
            print(f"参考翻译: {row['文本']}")
            print(f"OPUS翻译: {translated_text}\n")
    
    # 保存结果
    print(f"保存翻译结果到: {output_path}")
    malay_data.to_csv(output_path, index=False, encoding='utf-8')
    
    # 统计信息
    successful_translations = malay_data['answer'].str.len() > 0
    success_count = successful_translations.sum()
    print(f"\n翻译完成统计:")
    print(f"总计马来语数据: {len(malay_data)}")
    print(f"成功翻译: {success_count}")
    print(f"失败数量: {len(malay_data) - success_count}")
    print(f"成功率: {success_count/len(malay_data)*100:.2f}%")

def show_translation_examples(n=5):
    """显示翻译示例"""
    if not os.path.exists(output_path):
        print("输出文件不存在，请先运行翻译")
        return
    
    df = pd.read_csv(output_path)
    print(f"\n显示前{n}个翻译示例:")
    print("="*80)
    
    for i, (idx, row) in enumerate(df.head(n).iterrows()):
        print(f"示例 {i+1}:")
        print(f"编号: {row['编号']}")
        print(f"中文原文: {row['中文']}")
        print(f"参考马来语: {row['文本']}")
        print(f"OPUS翻译: {row['answer']}")
        print("-" * 40)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用OPUS模型翻译马来语数据')
    parser.add_argument('--action', choices=['translate', 'examples'], default='translate',
                       help='执行的操作: translate(翻译) 或 examples(显示示例)')
    parser.add_argument('--examples_count', type=int, default=5,
                       help='显示示例的数量')
    
    args = parser.parse_args()
    
    if args.action == 'translate':
        translate_malay_data()
    elif args.action == 'examples':
        show_translation_examples(args.examples_count)

if __name__ == "__main__":
    main() 