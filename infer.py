import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# 配置 PyTorch Dynamo
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # 增加缓存限制
torch._dynamo.config.suppress_errors = True  # 抑制编译错误
import pandas as pd
from tqdm import tqdm
import warnings
from fireredasr.models.fireredasr import FireRedAsr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from peft import PeftModel, PeftConfig
from argparse import Namespace
from datasets import Dataset
import numpy as np
torch.serialization.add_safe_globals([Namespace])

# 设置 PyTorch 优化
torch.set_float32_matmul_precision('high')  # 启用 TensorFloat32 提升性能
torch.backends.cudnn.benchmark = True  # 优化 cuDNN 性能
# 使用 bf16 精度进行推理，提供更好的数值稳定性

# 过滤掉特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='It is strongly recommended to pass the.*sampling_rate.*argument')

# 路径配置 - 使用相对路径
base_dir = os.getcwd()  # 当前工作目录

asr_model_path = os.path.join(base_dir, "models/FireRedASR-AED-L")
translate_model_path = os.path.join(base_dir, "models/GemmaX2-28-9B-v0.1")
translate_model_lora_path = os.path.join(base_dir, "models/pt/GemmaX2-28-9B-v0.1-pt-101/checkpoint-87000")

# NLLB模型路径配置
# 单模型方式：使用一个模型处理所有语言
nllb_single_model_path = os.path.join(base_dir, "models/ft/nllb-200-3.3B-ft-lora-102/checkpoint-3705")

# 多模型方式：为每种语言使用专用模型
nllb_multi_model_paths = {
    "Malay": os.path.join(base_dir, "models/ft/nllb-200-3.3B-ft-lora-203-Malay/checkpoint-2000"),
    "Thai": os.path.join(base_dir, "models/ft/nllb-200-3.3B-ft-lora-202-Thai/checkpoint-4200"), 
    "English": os.path.join(base_dir, "models/ft/nllb-200-3.3B-ft-lora-201-English/checkpoint-1600")
}

audio_dir = os.path.join(base_dir, "data")
input_path = os.path.join(base_dir, "data/text_data/testa.csv")
asr_output_path = os.path.join(base_dir, "output/testa_asr.csv")

# 语言映射
lang_map = {
    "马来语": "Malay",
    "泰语": "Thai",
    "英语": "English"
}

# NLLB语言代码映射
nllb_lang_map = {
    "马来语": "zsm_Latn",  # 马来语
    "泰语": "tha_Thai",    # 泰语
    "英语": "eng_Latn",    # 英语
    "Malay": "zsm_Latn",   # Malay (Standard) - 为了兼容lang_map的输出
    "Thai": "tha_Thai",    # Thai - 为了兼容lang_map的输出
    "English": "eng_Latn"  # English - 为了兼容lang_map的输出
}

# 全局变量存储模型
asr_model = None
translate_model = None
tokenizer = None
# NLLB相关全局变量
nllb_model = None
nllb_tokenizer = None
ml_translator = None  # 马来语翻译器
th_translator = None  # 泰语翻译器
en_translator = None  # 英语翻译器

# NLLB多模型全局变量
nllb_models = {}  # 存储多个模型
nllb_tokenizers = {}  # 存储多个tokenizer
use_multi_models = False  # 是否使用多模型模式

def init_asr_model():
    """初始化ASR模型"""
    global asr_model
    if asr_model is None:
        print("正在加载ASR模型...")
        asr_model = FireRedAsr.from_pretrained("aed", asr_model_path)
    return asr_model

def init_gemma_model():
    """初始化Gemma翻译模型"""
    global translate_model, tokenizer
    if translate_model is None:
        print("正在加载Gemma翻译模型...")
        tokenizer = AutoTokenizer.from_pretrained(translate_model_path)
        translate_model = AutoModelForCausalLM.from_pretrained(
            translate_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        translate_model = PeftModel.from_pretrained(translate_model, translate_model_lora_path)
        translate_model = translate_model.to("cuda")
        translate_model = translate_model.to(torch.bfloat16)
        print("Gemma模型已转换为 bf16 精度")
    return translate_model, tokenizer

def init_nllb_model(multi_model=False):
    """初始化NLLB翻译模型和翻译器
    
    Args:
        multi_model: 是否使用多模型模式
                    True - 为每种语言加载专用模型
                    False - 使用单个模型处理所有语言 (默认)
    """
    global nllb_model, nllb_tokenizer, ml_translator, th_translator, en_translator
    global nllb_models, nllb_tokenizers, use_multi_models
    
    use_multi_models = multi_model
    
    if multi_model:
        return init_nllb_multi_models()
    else:
        return init_nllb_single_model()

def init_nllb_single_model():
    """初始化单个NLLB模型处理所有语言"""
    global nllb_model, nllb_tokenizer, ml_translator, th_translator, en_translator
    
    if nllb_model is None:
        print("正在加载NLLB单模型...")
        try:
            # 检查本地路径是否存在且有效
            if nllb_single_model_path and os.path.exists(nllb_single_model_path):
                print(f"从本地路径加载: {nllb_single_model_path}")
                nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_single_model_path, use_auth_token=False, src_lang="zho_Hans")
                nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    nllb_single_model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_auth_token=False
                )
                print("本地NLLB单模型加载成功")
            else:
                print("本地模型路径不存在，从Hugging Face Hub加载NLLB模型...")
                # 从Hugging Face Hub加载
                nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", use_auth_token=False)
                nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "facebook/nllb-200-3.3B",
                    device_map="auto", 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_auth_token=False
                )
                print("Hugging Face NLLB模型加载成功")
        except Exception as e:
            print(f"NLLB模型加载失败: {e}")
            print("尝试加载distilled-600M版本...")
            try:
                # 尝试加载更小的版本
                nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=False)
                nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "facebook/nllb-200-distilled-600M",
                    device_map="auto", 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_auth_token=False
                )
                print("NLLB-600M模型加载成功")
            except Exception as e2:
                print(f"所有NLLB模型加载都失败: {e2}")
                return None, None
        
        # 创建统一的翻译器
        print("正在创建翻译器...")
        try:
            # 创建一个通用的翻译器
            nllb_translator = pipeline(
                'translation',
                model=nllb_model,
                tokenizer=nllb_tokenizer
            )
            print("NLLB翻译器创建成功")
            
            # 将通用翻译器赋值给各个语言的翻译器变量
            ml_translator = nllb_translator  # 马来语
            th_translator = nllb_translator  # 泰语  
            en_translator = nllb_translator  # 英语
            
            print("所有语言翻译器初始化完成")
            
        except Exception as e:
            print(f"创建翻译器过程中出错: {e}")
            print("将使用备用翻译方案")
    
    return nllb_model, nllb_tokenizer

def init_nllb_multi_models():
    """初始化多个NLLB模型，每种语言使用专用模型"""
    global nllb_models, nllb_tokenizers, ml_translator, th_translator, en_translator
    
    print("正在加载NLLB多模型...")
    
    for lang, model_path in nllb_multi_model_paths.items():
        if lang not in nllb_models:
            try:
                print(f"正在加载{lang}专用模型: {model_path}")
                
                if os.path.exists(model_path):
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=False, src_lang="zho_Hans")
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        use_auth_token=False
                    )
                    print(f"{lang}专用模型加载成功")
                else:
                    print(f"错误: {lang}专用模型路径不存在: {model_path}")
                    raise FileNotFoundError(f"模型路径不存在: {model_path}")
                
                nllb_models[lang] = model
                nllb_tokenizers[lang] = tokenizer
                
                # 创建专用翻译器
                translator = pipeline(
                    'translation',
                    model=model,
                    tokenizer=tokenizer
                )
                
                # 根据语言分配翻译器
                if lang == "Malay":
                    ml_translator = translator
                elif lang == "Thai":
                    th_translator = translator
                elif lang == "English":
                    en_translator = translator
                
                print(f"{lang}翻译器创建成功")
                
            except Exception as e:
                print(f"{lang}模型加载失败: {e}")
                print(f"模型路径: {model_path}")
                print(f"路径是否存在: {os.path.exists(model_path)}")
                if os.path.exists(model_path):
                    print(f"路径内容: {os.listdir(model_path) if os.path.isdir(model_path) else '不是目录'}")
                raise RuntimeError(f"无法加载{lang}模型: {e}")
    
    print("多模型初始化完成")
    return nllb_models, nllb_tokenizers

def asr_one(audio_path):
    """语音识别单个音频文件"""
    try:
        model = init_asr_model()
        results = model.transcribe(
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

def translate_one_gemma(text, lang):
    """使用Gemma模型翻译单个文本"""
    try:
        model, tok = init_gemma_model()
        prompt = f"Translate this from Chinese to {lang}:\nChinese: {text}\n{lang}:"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        if hasattr(inputs, 'input_ids'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id
                )
        
        translated_text = tok.decode(outputs[0], skip_special_tokens=True)
        
        # 清理缓存
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            return translated_text.split(f"{lang}:")[-1].strip()
        except:
            return translated_text
    except Exception as e:
        print(f"Gemma翻译文本出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ""

def translate_one_nllb(text, lang):
    """使用NLLB模型翻译单个文本 - 支持单模型和多模型模式"""
    try:
        # 根据模式选择不同的翻译策略
        if use_multi_models:
            return translate_one_nllb_multi(text, lang)
        else:
            return translate_one_nllb_single(text, lang)
            
    except Exception as e:
        print(f"NLLB翻译文本出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"[翻译错误] {text}"

def translate_one_nllb_single(text, lang):
    """使用单个NLLB模型翻译文本"""
    try:
        # 初始化模型（如果还没有初始化）
        model, tokenizer = init_nllb_model(multi_model=False)
        
        if model is None:
            print("NLLB模型未初始化")
            return f"[模型未初始化] {text}"
        
        # 根据目标语言选择对应的翻译器
        translator = None
        target_lang = None
        if lang in ["马来语", "Malay"]:
            translator = ml_translator
            target_lang = "zsm_Latn"
        elif lang in ["泰语", "Thai"]:
            translator = th_translator
            target_lang = "tha_Thai"
        elif lang in ["英语", "English"]:
            translator = en_translator
            target_lang = "eng_Latn"
        else:
            print(f"不支持的语言: {lang}")
            return f"[不支持的语言] {text}"
        
        # 如果翻译器未初始化，抛出错误
        if translator is None:
            raise RuntimeError(f"翻译器未初始化: {lang}")
        
        # 使用pipeline进行翻译，指定源语言和目标语言
        with torch.inference_mode():
            result = translator(text, src_lang="zho_Hans", tgt_lang=target_lang)
            
        # 提取翻译结果
        if isinstance(result, list) and len(result) > 0:
            translated_text = result[0]['translation_text']
        else:
            translated_text = str(result)
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"NLLB单模型翻译文本出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"[翻译错误] {text}"

def translate_one_nllb_multi(text, lang):
    """使用多个专用NLLB模型翻译文本"""
    try:
        # 确保多模型已初始化
        if not nllb_models:
            init_nllb_model(multi_model=True)
        
        # 根据目标语言选择对应的翻译器和模型
        translator = None
        target_lang = None
        model_lang = None
        
        if lang in ["马来语", "Malay"]:
            translator = ml_translator
            target_lang = "zsm_Latn"
            model_lang = "Malay"
        elif lang in ["泰语", "Thai"]:
            translator = th_translator
            target_lang = "tha_Thai"
            model_lang = "Thai"
        elif lang in ["英语", "English"]:
            translator = en_translator
            target_lang = "eng_Latn"
            model_lang = "English"
        else:
            print(f"不支持的语言: {lang}")
            return f"[不支持的语言] {text}"
        
        # 检查对应语言的模型是否加载成功
        if model_lang not in nllb_models or nllb_models[model_lang] is None:
            raise RuntimeError(f"{model_lang}专用模型未加载，模型路径可能不存在或加载失败")
        
        # 使用专用翻译器进行翻译
        if translator is None:
            raise RuntimeError(f"{model_lang}翻译器未初始化，无法进行翻译")
        
        # 使用pipeline进行翻译
        with torch.inference_mode():
            result = translator(text, src_lang="zho_Hans", tgt_lang=target_lang)
            
        # 提取翻译结果
        if isinstance(result, list) and len(result) > 0:
            translated_text = result[0]['translation_text']
        else:
            translated_text = str(result)
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"NLLB多模型翻译文本出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"[多模型翻译错误] {text}"

def translate_with_backup(text, lang, model, tokenizer):
    """备用翻译方案，直接使用模型"""
    try:
        # 设置源语言和目标语言
        source_lang = "zho_Hans"  # 简体中文
        target_lang = nllb_lang_map.get(lang, "eng_Latn")
        
        print(f"使用备用方案翻译: {text[:30]}... ({source_lang} -> {target_lang})")
        
        # 编码输入文本
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 设置强制的开始token为目标语言
                forced_bos_token_id = None
                if hasattr(tokenizer, 'lang_code_to_id') and target_lang in tokenizer.lang_code_to_id:
                    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
                
                # 生成翻译
                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": 200,
                    "do_sample": False,
                    "num_beams": 4,
                    "early_stopping": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
                
                if forced_bos_token_id is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
                
                generated_tokens = model.generate(**generate_kwargs)
        
        # 解码翻译结果
        input_length = inputs["input_ids"].shape[1]
        output_tokens = generated_tokens[0][input_length:]
        translated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        # 清理缓存
        del generated_tokens
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"备用翻译方案失败: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"[备用翻译失败] {text}"

def translate_one(text, lang, model_type="gemma"):
    """根据模型类型选择翻译方法"""
    if model_type == "nllb":
        return translate_one_nllb(text, lang)
    else:
        return translate_one_gemma(text, lang)

def asr():
    print("开始处理音频文件...")
    os.makedirs(os.path.dirname(asr_output_path), exist_ok=True)
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

def translate_efficient(model_type="gemma", multi_model=False, batch_size=None):
    """高效批量翻译主函数 - 使用Dataset和batch processing"""
    print(f"开始使用{model_type.upper()}模型高效批量翻译...")
    
    # 设置默认批大小
    if batch_size is None:
        batch_size = 16 if model_type == "gemma" else 32
    
    # 如果使用NLLB且启用多模型，设置全局标志
    if model_type == "nllb" and multi_model:
        global use_multi_models
        use_multi_models = True
        print("启用NLLB多模型模式 - 每种语言使用专用模型")
        # 预加载所有模型
        print("预加载所有专用模型...")
        init_nllb_model(multi_model=True)
    elif model_type == "nllb":
        print("使用NLLB单模型模式 - 一个模型处理所有语言")
    
    # 根据模型类型设置输出路径
    if model_type == "nllb":
        if multi_model:
            translate_output_path = os.path.join(base_dir, "output/nllb-multi/testa_translate.csv")
        else:
            translate_output_path = os.path.join(base_dir, "output/nllb/testa_translate.csv")
    else:
        translate_output_path = os.path.join(base_dir, f"output/{translate_model_path.split('/')[-2]}/testa_translate.csv")
    
    os.makedirs(os.path.dirname(translate_output_path), exist_ok=True)
    
    df = pd.read_csv(asr_output_path)
    
    # 检查已有结果
    if 'answer' not in df.columns:
        df['answer'] = None
    
    # 找出需要翻译的行
    need_translation = df['answer'].isna()
    if not need_translation.any():
        print("所有翻译都已完成！")
        return
    
    print(f"需要翻译 {need_translation.sum()} 行，使用批大小: {batch_size}")
    
    try:
        # 按语言分组进行批量翻译
        languages = df[need_translation].iloc[:, 3].unique()
        print(f"需要翻译的语言: {languages}")
        
        for target_lang in languages:
            print(f"\n=== 开始翻译 {target_lang} ===")
            
            # 获取该语言需要翻译的行
            lang_mask = (df.iloc[:, 3] == target_lang) & need_translation
            lang_indices = df[lang_mask].index.tolist()
            lang_texts = df.loc[lang_indices, df.columns[4]].tolist()  # 中文文本列
            
            if not lang_texts:
                continue
            
            print(f"该语言需要翻译 {len(lang_texts)} 个文本")
            
            # 转换语言名称
            mapped_lang = lang_map.get(target_lang, target_lang)
            
            # 批量翻译
            if model_type == "nllb":
                lang_results = translate_batch_nllb(lang_texts, mapped_lang, batch_size)
            else:
                lang_results = translate_batch_gemma(lang_texts, mapped_lang, batch_size)
            
            # 更新结果到DataFrame
            df.loc[lang_indices, 'answer'] = lang_results
            
            # 保存中间结果
            df.to_csv(translate_output_path, encoding='utf-8-sig', index=False)
            print(f"{target_lang} 翻译完成，已保存中间结果")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"批量翻译过程中出错: {str(e)}")
        # 保存已完成的结果
        df.to_csv(translate_output_path, encoding='utf-8-sig', index=False)
        raise

    # 最终保存
    df.to_csv(translate_output_path, encoding='utf-8-sig', index=False)
    print(f"所有翻译完成！结果已保存到：{translate_output_path}")
    
    # 统计翻译结果
    completed = df['answer'].notna().sum()
    total = len(df)
    print(f"翻译统计: {completed}/{total} 完成 ({completed/total*100:.1f}%)")
    
    # 显示一些翻译结果示例
    print("\n翻译结果示例:")
    sample_results = df[df['answer'].notna()].head(3)
    for _, row in sample_results.iterrows():
        print(f"中文: {row.iloc[4]}")
        print(f"目标语言: {row.iloc[3]}")
        print(f"翻译结果: {row['answer']}")
        print("-" * 50)

def translate(model_type="gemma", multi_model=False):
    """翻译主函数 - 保持原有的单条推理接口用于兼容性"""
    print("注意: 正在使用单条推理模式，效率较低。建议使用 --efficient 参数启用批量处理。")
    return translate_efficient(model_type, multi_model, batch_size=1)

def check_nllb_languages():
    """检查NLLB模型支持的语言代码"""
    try:
        init_nllb_model()
        
        print("NLLB翻译器状态检查:")
        print(f"马来语翻译器: {'已加载' if ml_translator else '未加载'}")
        print(f"泰语翻译器: {'已加载' if th_translator else '未加载'}")
        print(f"英语翻译器: {'已加载' if en_translator else '未加载'}")
        
        # 测试翻译功能
        test_text = "你好，世界！"
        print(f"\n测试翻译文本: {test_text}")
        
        if ml_translator:
            try:
                result = ml_translator(test_text)
                print(f"马来语翻译结果: {result}")
            except Exception as e:
                print(f"马来语翻译测试失败: {e}")
        
        if th_translator:
            try:
                result = th_translator(test_text)
                print(f"泰语翻译结果: {result}")
            except Exception as e:
                print(f"泰语翻译测试失败: {e}")
                
        if en_translator:
            try:
                result = en_translator(test_text)
                print(f"英语翻译结果: {result}")
            except Exception as e:
                print(f"英语翻译测试失败: {e}")
                
    except Exception as e:
        print(f"检查语言代码时出错: {e}")

def test_pipeline_creation():
    """测试pipeline创建，用于诊断问题"""
    try:
        print("=== 测试 Pipeline 创建 ===")
        
        # 初始化模型
        model, tokenizer = init_nllb_model()
        if model is None:
            print("模型未初始化，无法测试pipeline")
            return
        
        print(f"模型类型: {type(model)}")
        print(f"tokenizer类型: {type(tokenizer)}")
        
        # 检查tokenizer属性
        print("检查tokenizer属性:")
        if hasattr(tokenizer, 'lang_code_to_id'):
            print("  有 lang_code_to_id 属性")
            lang_codes = list(tokenizer.lang_code_to_id.keys())
            print(f"  语言代码数量: {len(lang_codes)}")
            # 检查目标语言是否存在
            target_langs = ['zho_Hans', 'zsm_Latn', 'tha_Thai', 'eng_Latn']
            for lang in target_langs:
                if lang in lang_codes:
                    print(f"  ✓ {lang}: 存在")
                else:
                    print(f"  ✗ {lang}: 不存在")
        else:
            print("  没有 lang_code_to_id 属性")
        
        # 尝试创建简单的pipeline（不指定语言）
        print("\n测试基础pipeline创建:")
        try:
            basic_pipeline = pipeline('translation', model=model, tokenizer=tokenizer)
            print("✓ 基础pipeline创建成功")
            
            # 测试翻译
            test_result = basic_pipeline("你好", src_lang="zho_Hans", tgt_lang="eng_Latn")
            print(f"✓ 基础翻译测试成功: {test_result}")
        except Exception as e:
            print(f"✗ 基础pipeline创建失败: {e}")
        
        # 测试带语言参数的pipeline创建
        print("\n测试带语言参数的pipeline创建:")
        test_langs = [
            ("zho_Hans", "eng_Latn", "英语"),
            ("zho_Hans", "zsm_Latn", "马来语"),
            ("zho_Hans", "tha_Thai", "泰语")
        ]
        
        for src, tgt, name in test_langs:
            try:
                test_pipeline = pipeline(
                    'translation',
                    model=model,
                    tokenizer=tokenizer,
                    src_lang=src,
                    tgt_lang=tgt,
                    device=next(model.parameters()).device
                )
                print(f"✓ {name}pipeline创建成功")
                
                # 测试翻译
                result = test_pipeline("你好")
                print(f"  翻译结果: {result}")
                
            except Exception as e:
                print(f"✗ {name}pipeline创建失败: {e}")
        
        # 测试新的翻译方案
        print("\n测试新的翻译方案:")
        test_texts = ["你好，世界！", "今天天气很好。", "谢谢你的帮助。"]
        test_langs = ["English", "Malay", "Thai"]
        
        for text in test_texts:
            for lang in test_langs:
                try:
                    result = translate_one_nllb(text, lang)
                    print(f"✓ {text} -> {lang}: {result}")
                except Exception as e:
                    print(f"✗ {text} -> {lang}: 失败 - {e}")
        
    except Exception as e:
        print(f"测试过程出错: {e}")

def translate_batch_nllb(texts, lang, batch_size=16):
    """批量翻译函数 - NLLB模型，使用Dataset提高效率"""
    try:
        # 根据模式选择翻译策略
        if use_multi_models:
            return translate_batch_nllb_multi(texts, lang, batch_size)
        else:
            return translate_batch_nllb_single(texts, lang, batch_size)
    except Exception as e:
        print(f"NLLB批量翻译出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [f"[翻译错误] {text}" for text in texts]

def translate_batch_nllb_single(texts, lang, batch_size=16):
    """使用单个NLLB模型批量翻译"""
    try:
        # 初始化模型
        model, tokenizer = init_nllb_model(multi_model=False)
        if model is None:
            print("NLLB模型未初始化")
            return [f"[模型未初始化] {text}" for text in texts]
        
        # 确定目标语言代码
        target_lang = nllb_lang_map.get(lang, "eng_Latn")
        source_lang = "zho_Hans"
        
        # 创建翻译pipeline，启用批处理
        # 注意：使用accelerate加载的模型不需要指定device
        translator = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            src_lang=source_lang,
            tgt_lang=target_lang
        )
        
        # 批量翻译
        print(f"开始批量翻译 {len(texts)} 个文本到 {lang} (批大小: {batch_size})")
        results = []
        
        # 分批处理以避免内存溢出
        for i in tqdm(range(0, len(texts), batch_size), desc="批量翻译"):
            batch_texts = texts[i:i+batch_size]
            
            with torch.inference_mode():
                batch_results = translator(batch_texts)
            
            # 提取翻译结果
            for result in batch_results:
                if isinstance(result, dict) and 'translation_text' in result:
                    results.append(result['translation_text'].strip())
                else:
                    results.append(str(result).strip())
            
            # 定期清理缓存
            if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"NLLB单模型批量翻译出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [f"[批量翻译错误] {text}" for text in texts]

def translate_batch_nllb_multi(texts, lang, batch_size=16):
    """使用多个专用NLLB模型批量翻译"""
    try:
        # 确保多模型已初始化
        if not nllb_models:
            init_nllb_model(multi_model=True)
        
        # 确定使用的模型和目标语言
        model_lang = None
        target_lang = None
        
        if lang in ["马来语", "Malay"]:
            model_lang = "Malay"
            target_lang = "zsm_Latn"
        elif lang in ["泰语", "Thai"]:
            model_lang = "Thai"
            target_lang = "tha_Thai"
        elif lang in ["英语", "English"]:
            model_lang = "English"
            target_lang = "eng_Latn"
        else:
            print(f"不支持的语言: {lang}")
            return [f"[不支持的语言] {text}" for text in texts]
        
        # 检查对应语言的模型是否加载成功
        if model_lang not in nllb_models or nllb_models[model_lang] is None:
            raise RuntimeError(f"{model_lang}专用模型未加载")
        
        model = nllb_models[model_lang]
        tokenizer = nllb_tokenizers[model_lang]
        source_lang = "zho_Hans"
        
        # 创建翻译pipeline
        # 注意：使用accelerate加载的模型不需要指定device
        translator = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            src_lang=source_lang,
            tgt_lang=target_lang
        )
        
        # 批量翻译
        print(f"使用{model_lang}专用模型批量翻译 {len(texts)} 个文本 (批大小: {batch_size})")
        results = []
        
        # 分批处理
        for i in tqdm(range(0, len(texts), batch_size), desc=f"{model_lang}批量翻译"):
            batch_texts = texts[i:i+batch_size]
            
            with torch.inference_mode():
                batch_results = translator(batch_texts)
            
            # 提取翻译结果
            for result in batch_results:
                if isinstance(result, dict) and 'translation_text' in result:
                    results.append(result['translation_text'].strip())
                else:
                    results.append(str(result).strip())
            
            # 定期清理缓存
            if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"NLLB多模型批量翻译出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [f"[多模型批量翻译错误] {text}" for text in texts]

def translate_batch_gemma(texts, lang, batch_size=8):
    """批量翻译函数 - Gemma模型，使用Dataset提高效率"""
    try:
        model, tok = init_gemma_model()
        results = []
        
        print(f"使用Gemma模型批量翻译 {len(texts)} 个文本到 {lang} (批大小: {batch_size})")
        
        # 分批处理
        for i in tqdm(range(0, len(texts), batch_size), desc="Gemma批量翻译"):
            batch_texts = texts[i:i+batch_size]
            
            # 创建批量prompt
            prompts = [f"Translate this from Chinese to {lang}:\nChinese: {text}\n{lang}:" for text in batch_texts]
            
            # 批量编码
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            
            with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True,
                        pad_token_id=tok.eos_token_id
                    )
            
            # 批量解码
            for j, output in enumerate(outputs):
                translated_text = tok.decode(output, skip_special_tokens=True)
                try:
                    result = translated_text.split(f"{lang}:")[-1].strip()
                    results.append(result)
                except:
                    results.append(translated_text)
            
            # 清理缓存
            del outputs
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"Gemma批量翻译出错: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [f"[翻译错误] {text}" for text in texts]

def show_model_status():
    """显示当前模型状态和配置"""
    print("=== 模型配置状态 ===")
    
    # ASR模型状态
    print(f"ASR模型路径: {asr_model_path}")
    print(f"ASR模型状态: {'已加载' if asr_model is not None else '未加载'}")
    
    # Gemma模型状态
    print(f"\nGemma模型路径: {translate_model_path}")
    print(f"Gemma LoRA路径: {translate_model_lora_path}")
    print(f"Gemma模型状态: {'已加载' if translate_model is not None else '未加载'}")
    
    # NLLB模型状态
    print(f"\n=== NLLB模型配置 ===")
    print(f"使用多模型模式: {'是' if use_multi_models else '否'}")
    
    if use_multi_models:
        print("多模型路径配置:")
        for lang, path in nllb_multi_model_paths.items():
            status = "已加载" if lang in nllb_models and nllb_models[lang] is not None else "未加载"
            exists = "存在" if os.path.exists(path) else "不存在"
            print(f"  {lang}: {path} (文件{exists}, 模型{status})")
        
        print("\n翻译器状态:")
        print(f"  马来语翻译器: {'已创建' if ml_translator is not None else '未创建'}")
        print(f"  泰语翻译器: {'已创建' if th_translator is not None else '未创建'}")
        print(f"  英语翻译器: {'已创建' if en_translator is not None else '未创建'}")
    else:
        print(f"单模型路径: {nllb_single_model_path}")
        print(f"单模型文件存在: {'是' if os.path.exists(nllb_single_model_path) else '否'}")
        print(f"NLLB模型状态: {'已加载' if nllb_model is not None else '未加载'}")
        print(f"NLLB翻译器状态: {'已创建' if ml_translator is not None else '未创建'}")
    
    # 输出路径配置
    print(f"\n=== 输出路径配置 ===")
    print(f"ASR输出路径: {asr_output_path}")
    gemma_output_dir = translate_model_path.split("/")[-2]
    print(f"Gemma翻译输出路径: {os.path.join(base_dir, f'output/{gemma_output_dir}/testa_translate.csv')}")
    print(f"NLLB单模型输出路径: {os.path.join(base_dir, 'output/nllb/testa_translate.csv')}")
    print(f"NLLB多模型输出路径: {os.path.join(base_dir, 'output/nllb-multi/testa_translate.csv')}")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='语音识别和翻译推理工具')
    parser.add_argument('--task', choices=['asr', 'translate', 'all', 'test', 'test_batch', 'status'], default='all',
                      help='选择任务: asr(语音识别), translate(翻译), all(全部), test(测试pipeline), test_batch(测试批量翻译), status(显示模型状态)')
    parser.add_argument('--model', choices=['gemma', 'nllb'], default='gemma',
                      help='选择翻译模型: gemma, nllb')
    parser.add_argument('--multi-model', action='store_true',
                      help='当使用NLLB时，启用多模型模式 (每种语言使用专用模型)')
    parser.add_argument('--efficient', action='store_true', default=True,
                      help='启用高效批量处理模式 (默认启用)')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='批处理大小 (默认: Gemma=8, NLLB=16)')
    
    args = parser.parse_args()
    
    if args.task == 'test':
        test_pipeline_creation()
        return
    
    if args.task == 'status':
        show_model_status()
        return
    
    if args.task in ['asr', 'all']:
        asr()
    
    if args.task in ['translate', 'all']:
        # 传递multi_model参数，只在使用nllb时有效
        multi_model = getattr(args, 'multi_model', False) and args.model == 'nllb'
        
        if args.efficient:
            translate_efficient(args.model, multi_model, args.batch_size)
        else:
            translate(args.model, multi_model)

if __name__ == "__main__":
    main() 