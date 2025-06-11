import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 配置 PyTorch Dynamo
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # 增加缓存限制
torch._dynamo.config.suppress_errors = True  # 抑制编译错误
import pandas as pd
from tqdm import tqdm
import warnings
from fireredasr.models.fireredasr import FireRedAsr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
from argparse import Namespace
torch.serialization.add_safe_globals([Namespace])

# 过滤掉特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='It is strongly recommended to pass the.*sampling_rate.*argument')

# 路径配置
asr_model_path = "/home/zbtrs/lz/trans/models/FireRedASR-AED-L"
translate_model_path = "/home/zbtrs/lz/trans/models/GemmaX2-28-2B-v0.1"
translate_model_lora_path = "/home/zbtrs/lz/trans/models/GemmaX2-28-2B-v0.1-finetuned-10/checkpoint-6084"

audio_dir = "/home/zbtrs/lz/trans/data"
input_path = "/home/zbtrs/lz/trans/data/text_data/testa.csv"
asr_output_path = "/home/zbtrs/lz/trans/output/v2_4/testa_asr.csv"
translate_output_path = "/home/zbtrs/lz/trans/output/v2_4/testa_translate.csv"

os.makedirs(os.path.dirname(asr_output_path), exist_ok=True)

lang_map = {
    "马来语": "malay",
    "泰语": "thai",
    "英语": "english"
}

# 初始化模型
asr_model = FireRedAsr.from_pretrained("aed", asr_model_path)
tokenizer = AutoTokenizer.from_pretrained(translate_model_path)
translate_model = AutoModelForCausalLM.from_pretrained(
    translate_model_path,
    device_map="auto"
)
translate_model = PeftModel.from_pretrained(translate_model, translate_model_lora_path)
translate_model = translate_model.to("cuda")

def asr_one(audio_path):
    try:
        results = asr_model.transcribe(
            ["current_audio"],
            [audio_path],
            {
                "use_gpu": 1,
                "beam_size": 3,
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.0,
                "aed_length_penalty": 0.0,
                "eos_penalty": 1.0
            }
        )
        return results[0]["text"] if results else ""
    except Exception as e:
        print(f"处理音频文件出错 {audio_path}: {str(e)}")
        return ""

def translate_one(text, lang):
    try:
        prompt = f"Translate this from Chinese to {lang}:\nChinese: {text}\n{lang}:"
        inputs = tokenizer(prompt, return_tensors="pt").to(translate_model.device)
        
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
            outputs = translate_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True
            )
        
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 清理缓存
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 提取翻译结果
        try:
            return translated_text.split(f"{lang}:")[-1].strip()
        except:
            return translated_text
    except Exception as e:
        print(f"翻译文本出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""

def asr():
    print("开始处理音频文件...")
    df = pd.read_csv(input_path)
    results = []
    previous_audio_path = None
    previous_result = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频文件"):
        if idx and idx % 10 == 0:
            df.loc[0:len(results)-1, '中文'] = results
            df.to_csv(asr_output_path, encoding='utf-8-sig', index=False)

        audio_path = os.path.join(audio_dir, row.iloc[6][1:])

        if pd.notna(df.loc[idx, '中文']):
            results.append(df.loc[idx, '中文'])
            continue
        
        if audio_path == previous_audio_path and previous_result is not None:
            results.append(previous_result)
            continue
        
        result = asr_one(audio_path)
        results.append(result)

        previous_audio_path = audio_path
        previous_result = result

    df.loc[0:len(results)-1, '中文'] = results
    df.to_csv(asr_output_path, encoding='utf-8-sig', index=False)
    print(f"处理完成！结果已保存到：{asr_output_path}")

def translate():
    print("开始处理翻译文件...")
    df = pd.read_csv(asr_output_path)
    results = []
    
    # 设置较小的批处理大小以避免内存问题
    batch_size = 1
    
    try:
        for idx in tqdm(range(0, len(df), batch_size), desc="处理翻译文件"):
            batch_end = min(idx + batch_size, len(df))
            batch_df = df[idx:batch_end]
            
            for _, row in batch_df.iterrows():
                if pd.notna(row['answer']):
                    results.append(row['answer'])
                    continue
                
                result = translate_one(row.iloc[4], lang_map[row.iloc[3]])
                results.append(result)
                
                # 定期保存结果
                if len(results) % 5 == 0:  # 更频繁地保存结果
                    df.loc[0:len(results)-1, 'answer'] = results
                    df.to_csv(translate_output_path, encoding='utf-8-sig', index=False)
            
            # 每个批次后清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"翻译过程中出错: {str(e)}")
        # 保存已完成的结果
        if results:
            df.loc[0:len(results)-1, 'answer'] = results
            df.to_csv(translate_output_path, encoding='utf-8-sig', index=False)
        raise

    df.loc[0:len(results)-1, 'answer'] = results
    df.to_csv(translate_output_path, encoding='utf-8-sig', index=False)
    print(f"处理完成！结果已保存到：{translate_output_path}")

if __name__ == "__main__":
    # asr()
    translate() 