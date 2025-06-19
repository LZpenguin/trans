import os
# 设置GPU和代理
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'

import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    NllbTokenizer,  # 添加NLLB专用tokenizer
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, concatenate_datasets
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import warnings
import logging
import wandb
import numpy as np
import random
import hashlib
from dataset import TranslationDataset
from nltk.translate.bleu_score import sentence_bleu
from comet import download_model, load_from_checkpoint

# 尝试导入泰语分词工具
try:
    from pythainlp import word_tokenize as thai_tokenize
    THAI_TOKENIZER_AVAILABLE = True
except ImportError:
    THAI_TOKENIZER_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings('ignore')

# 添加主进程检查函数
def is_main_process():
    """检查是否为主进程"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

def log_info(message):
    """只在主进程打印日志"""
    if is_main_process():
        logger.info(message)

def log_warning(message):
    """只在主进程打印警告"""
    if is_main_process():
        logger.warning(message)

def log_error(message):
    """只在主进程打印错误"""
    if is_main_process():
        logger.error(message)

# 设置随机种子函数
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# 添加多语言分词函数
def tokenize_for_bleu(text, language):
    """根据语言类型进行适当的分词"""
    if language in ["Thai", "泰语"]:
        if THAI_TOKENIZER_AVAILABLE:
            # 使用泰语分词工具
            try:
                tokens = thai_tokenize(text, engine='newmm')
                return tokens
            except Exception as e:
                log_warning(f"泰语分词失败，使用字符级分词: {e}")
                # 如果分词失败，使用字符级分词（去除空格）
                return list(text.replace(" ", ""))
        else:
            # 如果没有安装pythainlp，使用字符级分词
            log_warning("未安装pythainlp，泰语使用字符级分词")
            return list(text.replace(" ", ""))
    elif language in ["English", "英语"]:
        # 英语使用空格分词
        return text.split()
    elif language in ["Malay", "马来语"]:
        # 马来语也使用空格分词
        return text.split()
    else:
        # 默认使用空格分词
        return text.split()

examples_data = []

# 配置参数
class TrainingConfig:
    # 获取当前工作目录的上级目录作为基础路径
    base_dir = os.path.join(os.getcwd(), "..")
    
    model_path = os.path.join(base_dir, "models/nllb-200-3.3B")
    # 统一训练文件列表，支持parquet和csv格式
    train_files = [
        os.path.join(base_dir, "data/text_data/train.csv")
    ]
    dev_file = os.path.join(base_dir, "data/text_data/dev.csv")

    # 指定训练的目标语言 - 可选值: "English", "Malay", "Thai", "英语", "马来语", "泰语", "all"
    target_language = "all"  # 设置为具体语言名称可以只训练该语言，设置为"all"训练所有语言
    
    # LoRA配置
    use_lora = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    
    # 根据训练模式和目标语言设置输出目录
    training_method = "ft"
    training_mode = "lora" if use_lora else "full"
    training_idx = 102
    
    # 根据目标语言调整输出目录
    lang_suffix = f"-{target_language}" if target_language != "all" else ""
    output_dir = os.path.join(base_dir, f"models/{training_method}/{model_path.split('/')[-1]}-{training_method}-{training_mode}-{training_idx}{lang_suffix}")
    
    # 训练配置
    batch_size = 32
    learning_rate = 1e-5
    num_epochs = 10
    max_source_length = 128
    max_target_length = 128
    gradient_accumulation_steps = 1
    save_total_limit = 100
    eval_steps = 200  # 手动指定评估步数
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    lr_scheduler_type = "cosine"  # 学习率调度器类型
    
    # 根据目标语言调整wandb配置
    wandb_project = f"nllb-translation"
    wandb_name = f"nllb-translation-{training_method}-{training_mode}-{training_idx}{lang_suffix}"
    
    # 添加随机种子
    seed = 42
    
    # 测试模式
    test_only = False  # 设置为True时只运行推理测试
    
    # 推理示例配置
    num_inference_examples = 10  # 每次评估时推送到wandb的推理示例数量
    
    # NLLB语言代码映射
    nllb_lang_map = {
        "马来语": "zsm_Latn",
        "泰语": "tha_Thai", 
        "英语": "eng_Latn",
        "Malay": "zsm_Latn",
        "Thai": "tha_Thai",
        "English": "eng_Latn",
        "中文": "zho_Hans",
        "Chinese": "zho_Hans"
    }

def get_processed_data_cache_path(config: 'TrainingConfig') -> tuple:
    """获取已处理数据的缓存路径"""
    train_files_str = "_".join(sorted(config.train_files))
    train_files_hash = hashlib.md5(train_files_str.encode()).hexdigest()
    dev_file_hash = hashlib.md5(config.dev_file.encode()).hexdigest()
    target_lang_hash = hashlib.md5(config.target_language.encode()).hexdigest()
    cache_id = f"{config.model_path}_{config.max_source_length}_{config.max_target_length}_{config.seed}_{train_files_hash}_{dev_file_hash}_{target_lang_hash}_nllb_processed"
    cache_name = hashlib.md5(cache_id.encode()).hexdigest()
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/nllb_processed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache = os.path.join(cache_dir, f"{cache_name}_train")
    dev_cache = os.path.join(cache_dir, f"{cache_name}_dev")
    
    return train_cache, dev_cache

def prepare_data(config):
    """根据文件扩展名自动选择对应的数据加载器，并根据配置过滤目标语言"""
    train_datasets = []
    
    # 遍历所有训练文件
    for train_file in config.train_files:
        log_info(f"加载训练数据: {train_file}")
        
        if train_file.endswith('.parquet'):
            dataset_loader = TranslationDataset(data_type="parquet", double=False)
        elif train_file.endswith('.csv'):
            dataset_loader = TranslationDataset(data_type="csv", double=False)
        else:
            raise ValueError(f"不支持的文件格式: {train_file}")
        
        dataset = dataset_loader.load_data(train_file)
        
        # 根据配置过滤目标语言
        if config.target_language != "all":
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: x['target_lang'] == config.target_language)
            log_info(f"过滤目标语言 '{config.target_language}': {original_size} -> {len(dataset)} 样本")
        
        train_datasets.append(dataset)
        log_info(f"数据集大小: {len(dataset)}")
    
    # 合并所有训练数据集
    train_dataset = concatenate_datasets(train_datasets)
    log_info(f"合并后的训练数据集大小: {len(train_dataset)}")
    
    # 加载开发集数据
    log_info(f"加载开发集数据: {config.dev_file}")
    if config.dev_file.endswith('.csv'):
        dev_dataset_loader = TranslationDataset(data_type="csv", double=False)
    elif config.dev_file.endswith('.parquet'):
        dev_dataset_loader = TranslationDataset(data_type="parquet", double=False)
    else:
        raise ValueError(f"不支持的开发集文件格式: {config.dev_file}")
    
    dev_dataset = dev_dataset_loader.load_data(config.dev_file)
    
    # 根据配置过滤开发集的目标语言
    if config.target_language != "all":
        original_dev_size = len(dev_dataset)
        dev_dataset = dev_dataset.filter(lambda x: x['target_lang'] == config.target_language)
        log_info(f"过滤开发集目标语言 '{config.target_language}': {original_dev_size} -> {len(dev_dataset)} 样本")
    
    log_info(f"开发集数据大小: {len(dev_dataset)}")
    
    return train_dataset, dev_dataset

def prepare_model_and_tokenizer(config):
    """加载NLLB模型和tokenizer"""
    # 加载NLLB tokenizer - 根据目标语言选择性创建tokenizer
    tokenizers = {}
    
    # 语言到tokenizer键的映射
    lang_to_tokenizer_key = {
        "English": "eng",
        "Malay": "zsm", 
        "Thai": "tha",
        "英语": "eng",
        "马来语": "zsm",
        "泰语": "tha"
    }
    
    if config.target_language == "all":
        # 训练所有语言时，创建所有tokenizer
        tokenizers['eng'] = AutoTokenizer.from_pretrained(
            config.model_path,
            src_lang="zho_Hans",
            tgt_lang="eng_Latn"
        )
        log_info(f"创建中文->英语 tokenizer: {tokenizers['eng'].src_lang} -> {tokenizers['eng'].tgt_lang}")
        
        tokenizers['zsm'] = AutoTokenizer.from_pretrained(
            config.model_path,
            src_lang="zho_Hans",
            tgt_lang="zsm_Latn"
        )
        log_info(f"创建中文->马来语 tokenizer: {tokenizers['zsm'].src_lang} -> {tokenizers['zsm'].tgt_lang}")
        
        tokenizers['tha'] = AutoTokenizer.from_pretrained(
            config.model_path,
            src_lang="zho_Hans",
            tgt_lang="tha_Thai"
        )
        log_info(f"创建中文->泰语 tokenizer: {tokenizers['tha'].src_lang} -> {tokenizers['tha'].tgt_lang}")
        
        # 默认使用英语tokenizer进行模型加载
        tokenizer = tokenizers['eng']
    else:
        # 只训练单一语言时，只创建对应的tokenizer
        tokenizer_key = lang_to_tokenizer_key.get(config.target_language, "eng")
        
        if tokenizer_key == "eng":
            tgt_lang_code = "eng_Latn"
        elif tokenizer_key == "zsm":
            tgt_lang_code = "zsm_Latn"
        elif tokenizer_key == "tha":
            tgt_lang_code = "tha_Thai"
        else:
            tgt_lang_code = "eng_Latn"  # 默认
        
        tokenizers[tokenizer_key] = AutoTokenizer.from_pretrained(
            config.model_path,
            src_lang="zho_Hans",
            tgt_lang=tgt_lang_code
        )
        log_info(f"创建中文->{config.target_language} tokenizer: {tokenizers[tokenizer_key].src_lang} -> {tokenizers[tokenizer_key].tgt_lang}")
        
        # 使用该语言的tokenizer进行模型加载
        tokenizer = tokenizers[tokenizer_key]
    
    # 统一使用bfloat16数据类型
    torch_dtype = torch.bfloat16
    
    # 加载NLLB Seq2Seq模型
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    
    # 根据开关决定是否应用LoRA
    if config.use_lora:
        # 配置LoRA for Seq2Seq模型
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        log_info("使用LoRA训练模式")
    else:
        model.gradient_checkpointing_enable()
        log_info("使用全量训练模式，已启用梯度检查点")
    
    log_info(f"模型数据类型: {torch_dtype}")
    
    return model, tokenizers

def tokenize_function(examples, tokenizers, config):
    """NLLB模型的tokenization函数 - 使用预初始化的多个tokenizer"""
    # 获取源文本、目标文本和目标语言
    source_texts = examples['source_text']
    target_texts = examples['target_text']
    target_langs = examples['target_lang']
    
    # 语言代码到tokenizer键的映射
    lang_to_tokenizer_key = {
        "English": "eng",
        "Malay": "zsm", 
        "Thai": "tha",
        "英语": "eng",
        "马来语": "zsm",
        "泰语": "tha"
    }
    
    # 选择用于编码输入的tokenizer（所有tokenizer的src_lang都是中文，所以结果相同）
    # 优先使用英语tokenizer，如果不存在则使用第一个可用的tokenizer
    if 'eng' in tokenizers:
        input_tokenizer = tokenizers['eng']
    else:
        # 单语言训练时，使用唯一的tokenizer
        input_tokenizer = list(tokenizers.values())[0]
    
    model_inputs = input_tokenizer(
        source_texts,
        max_length=config.max_source_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 为每个样本根据目标语言选择对应的tokenizer编码标签
    labels_list = []
    for i, (tgt_text, tgt_lang) in enumerate(zip(target_texts, target_langs)):
        # 选择对应的tokenizer
        tokenizer_key = lang_to_tokenizer_key.get(tgt_lang, "eng")
        
        # 如果目标tokenizer不存在，使用第一个可用的tokenizer（单语言训练情况）
        if tokenizer_key in tokenizers:
            selected_tokenizer = tokenizers[tokenizer_key]
        else:
            selected_tokenizer = list(tokenizers.values())[0]
        
        # 编码目标文本
        labels = selected_tokenizer(
            tgt_text,
            max_length=config.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels_list.append(labels["input_ids"].squeeze())
    
    # 将所有labels合并
    labels_input_ids = torch.stack(labels_list)
    
    # 将padding token替换为-100（表示在loss计算中忽略）
    labels_input_ids[labels_input_ids == input_tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels_input_ids
    
    return model_inputs

class TranslationEvaluator:
    """翻译评估器，用于评估模型性能"""
    
    def __init__(self, config, tokenizers, accelerator):
        self.config = config
        self.tokenizers = tokenizers
        self.accelerator = accelerator
        self._comet_model = None
        
    def _load_comet_model(self):
        """加载COMET模型"""
        if self._comet_model is None:
            try:
                local_model_path = "/mnt/gold/lz/trans/models/XCOMET-XL/checkpoints/model.ckpt"
                self._comet_model = load_from_checkpoint(local_model_path)
            except Exception as e:
                log_warning(f"COMET模型加载失败: {e}")
                self._comet_model = None
        return self._comet_model
        
    def evaluate_model(self, model, original_dev_data, num_samples=500):
        """评估模型性能"""
        if not self.accelerator.is_main_process:
            return 0.0, 0.0, 0.0
            
        log_info("开始计算翻译评价指标...")
        
        # 按目标语言分组，确保三种语言均衡抽样
        lang_groups = {
            "English": [], "英语": [],
            "Malay": [], "马来语": [],  
            "Thai": [], "泰语": []
        }
        
        # 将数据按语言分组
        for idx, example in enumerate(original_dev_data):
            target_lang = example['target_lang']
            if target_lang in lang_groups:
                lang_groups[target_lang].append(idx)
        
        # 合并同类语言（中英文表示）
        english_indices = lang_groups["English"] + lang_groups["英语"]
        malay_indices = lang_groups["Malay"] + lang_groups["马来语"]
        thai_indices = lang_groups["Thai"] + lang_groups["泰语"]
        
        # 计算每种语言应该抽取的样本数
        total_available = len(english_indices) + len(malay_indices) + len(thai_indices)
        if total_available == 0:
            log_warning("未找到有效的目标语言数据")
            return 0.0, 0.0, 0.0
            
        sample_size = min(num_samples, total_available)
        samples_per_lang = sample_size // 3
        remaining_samples = sample_size % 3
        
        log_info(f"语言分布 - 英语: {len(english_indices)}, 马来语: {len(malay_indices)}, 泰语: {len(thai_indices)}")
        log_info(f"每种语言抽样: {samples_per_lang}, 剩余样本: {remaining_samples}")
        
        # 检查泰语分词工具状态
        if is_main_process():
            if THAI_TOKENIZER_AVAILABLE:
                log_info("已加载泰语分词工具 pythainlp")
            else:
                log_warning("未安装 pythainlp，泰语将使用字符级分词")
        
        # 存储每种语言的评估数据
        language_data = {
            "英语": {"indices": english_indices, "sources": [], "references": [], "predictions": [], "target_langs": []},
            "马来语": {"indices": malay_indices, "sources": [], "references": [], "predictions": [], "target_langs": []},
            "泰语": {"indices": thai_indices, "sources": [], "references": [], "predictions": [], "target_langs": []}
        }
        
        # 从每种语言中抽样并收集数据
        lang_info = [
            ("英语", english_indices),
            ("马来语", malay_indices), 
            ("泰语", thai_indices)
        ]
        
        all_sources = []
        all_references = []
        all_target_langs = []
        all_predictions = []
        
        for i, (lang_name, indices) in enumerate(lang_info):
            if not indices:
                log_warning(f"{lang_name}数据为空，跳过")
                continue
                
            # 基础抽样数量
            current_samples = samples_per_lang
            # 分配剩余样本
            if i < remaining_samples:
                current_samples += 1
                
            # 实际可抽样数量
            actual_samples = min(current_samples, len(indices))
            
            if actual_samples > 0:
                sampled_indices = random.sample(indices, actual_samples)
                log_info(f"{lang_name}: 抽样 {actual_samples} 个样本")
                
                # 收集该语言的数据
                lang_sources = []
                lang_references = []
                lang_target_langs = []
                
                for idx in sampled_indices:
                    example = original_dev_data[idx]
                    lang_sources.append(example['source_text'])
                    lang_references.append(example['target_text'])
                    lang_target_langs.append(example['target_lang'])
                
                # 批量推理该语言的数据
                lang_predictions = self.inference_batch(model, lang_sources, lang_target_langs, batch_size=16)
                
                # 存储该语言的数据
                language_data[lang_name]["sources"] = lang_sources
                language_data[lang_name]["references"] = lang_references
                language_data[lang_name]["predictions"] = lang_predictions
                language_data[lang_name]["target_langs"] = lang_target_langs
                
                # 添加到总体数据中
                all_sources.extend(lang_sources)
                all_references.extend(lang_references)
                all_target_langs.extend(lang_target_langs)
                all_predictions.extend(lang_predictions)
        
        # 分别计算每种语言的指标
        language_metrics = {}
        
        for lang_name, data in language_data.items():
            if not data["sources"]:  # 如果该语言没有数据，跳过
                continue
                
            sources = data["sources"]
            references = data["references"]
            predictions = data["predictions"]
            
            # 计算该语言的BLEU2分数
            bleu_scores = []
            for i, (ref, pred, tgt_lang) in enumerate(zip(references, predictions, data["target_langs"])):
                try:
                    ref_tokens = tokenize_for_bleu(ref, tgt_lang)
                    pred_tokens = tokenize_for_bleu(pred, tgt_lang)
                    
                    # # 对于泰语，打印前几个示例的分词结果进行调试
                    # if tgt_lang in ["Thai", "泰语"] and i < 3 and is_main_process():
                    #     log_info(f"泰语分词示例 {i+1}:")
                    #     log_info(f"  原文: {ref}")
                    #     log_info(f"  预测: {pred}")
                    #     log_info(f"  原文分词: {ref_tokens[:10]}..." if len(ref_tokens) > 10 else f"  原文分词: {ref_tokens}")
                    #     log_info(f"  预测分词: {pred_tokens[:10]}..." if len(pred_tokens) > 10 else f"  预测分词: {pred_tokens}")
                    
                    if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                        bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0))
                        bleu_scores.append(bleu)
                    else:
                        bleu_scores.append(0.0)
                        if tgt_lang in ["Thai", "泰语"] and i < 3 and is_main_process():
                            log_info(f"  分词结果为空，BLEU2分数: 0.0000")
                except Exception as e:
                    log_warning(f"BLEU计算失败 ({tgt_lang}): {e}")
                    bleu_scores.append(0.0)
            
            lang_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
            
            # 计算该语言的COMET分数
            lang_comet = 0.0
            try:
                comet_model = self._load_comet_model()
                if comet_model is not None:
                    comet_data = []
                    for src, ref, pred in zip(sources, references, predictions):
                        comet_data.append({
                            "src": src,
                            "mt": pred,
                            "ref": ref
                        })
                    
                    comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
                    lang_comet = np.mean(comet_output.scores)
            except Exception as e:
                if is_main_process():
                    logger.error(f"{lang_name} COMET分数计算失败: {e}")
                lang_comet = 0.0
            
            # 计算该语言的最终分数
            lang_final = lang_bleu * 0.4 + lang_comet * 0.6
            
            # 存储该语言的指标
            language_metrics[lang_name] = {
                "bleu": lang_bleu,
                "comet": lang_comet,
                "final": lang_final
            }
            
            log_info(f"{lang_name} - BLEU2: {lang_bleu:.4f}, COMET: {lang_comet:.4f}, Final: {lang_final:.4f} (样本数: {len(sources)})")
        
        # 计算总体平均分数 - 使用各语言分数的加权平均
        if language_metrics:
            # 计算加权平均BLEU分数
            total_bleu_weighted = 0.0
            total_comet_weighted = 0.0
            total_samples = 0
            
            for lang_name, metrics in language_metrics.items():
                lang_samples = len(language_data[lang_name]["sources"])
                if lang_samples > 0:
                    total_bleu_weighted += metrics["bleu"] * lang_samples
                    total_comet_weighted += metrics["comet"] * lang_samples
                    total_samples += lang_samples
            
            # 计算加权平均值
            avg_bleu = total_bleu_weighted / total_samples if total_samples > 0 else 0.0
            comet_score = total_comet_weighted / total_samples if total_samples > 0 else 0.0
            
            # 计算总体最终分数
            final_score = avg_bleu * 0.4 + comet_score * 0.6
        else:
            avg_bleu = 0.0
            comet_score = 0.0
            final_score = 0.0
        
        log_info(f"总体 - BLEU2: {avg_bleu:.4f}, COMET: {comet_score:.4f}, Final: {final_score:.4f}")
        
        # 记录各语言指标到wandb
        if is_main_process():
            try:
                import wandb
                if wandb.run:
                    wandb_metrics = {}
                    for lang_name, metrics in language_metrics.items():
                        wandb_metrics[f"eval/{lang_name}_bleu2"] = metrics["bleu"]
                        wandb_metrics[f"eval/{lang_name}_comet"] = metrics["comet"]
                        wandb_metrics[f"eval/{lang_name}_final"] = metrics["final"]
                    
                    # 添加总体指标
                    wandb_metrics["eval/overall_bleu2"] = avg_bleu
                    wandb_metrics["eval/overall_comet"] = comet_score
                    wandb_metrics["eval/overall_final"] = final_score
                    
                    wandb.log(wandb_metrics)
                    log_info(f"已将分语言评估指标记录到 wandb")
            except Exception as e:
                log_warning(f"记录分语言评估指标到 wandb 失败: {e}")
        
        return avg_bleu, comet_score, final_score
    
    def inference_batch(self, model, source_texts, target_langs, batch_size=8):
        """批量推理"""
        all_predictions = []
        
        lang_to_tokenizer_key = {
            "English": "eng",
            "Malay": "zsm", 
            "Thai": "tha",
            "英语": "eng",
            "马来语": "zsm",
            "泰语": "tha"
        }
        
        # 按目标语言分组
        lang_groups = {}
        for i, (src_text, tgt_lang) in enumerate(zip(source_texts, target_langs)):
            tokenizer_key = lang_to_tokenizer_key.get(tgt_lang, "eng")
            if tokenizer_key not in lang_groups:
                lang_groups[tokenizer_key] = []
            lang_groups[tokenizer_key].append((i, src_text, tgt_lang))
        
        predictions = [None] * len(source_texts)
        
        try:
            with torch.inference_mode():
                for tokenizer_key, group_data in lang_groups.items():
                    selected_tokenizer = self.tokenizers[tokenizer_key]
                    target_lang_code = selected_tokenizer.tgt_lang
                    
                    indices = [item[0] for item in group_data]
                    texts = [item[1] for item in group_data]
                    
                    num_batches = (len(texts) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        batch_start = batch_idx * batch_size
                        batch_end = min(batch_start + batch_size, len(texts))
                        batch_texts = texts[batch_start:batch_end]
                        batch_indices = indices[batch_start:batch_end]
                        
                        inputs = selected_tokenizer(
                            batch_texts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=self.config.max_source_length
                        ).to(model.device)
                        
                        generate_kwargs = {
                            "max_length": self.config.max_target_length,
                            "do_sample": False,
                            "num_beams": 4,
                            "early_stopping": True,
                            "forced_bos_token_id": selected_tokenizer.convert_tokens_to_ids(target_lang_code)
                        }
                        
                        translated_tokens = model.generate(**inputs, **generate_kwargs)
                        
                        batch_translations = selected_tokenizer.batch_decode(
                            translated_tokens, skip_special_tokens=True
                        )
                        
                        for idx, translation in zip(batch_indices, batch_translations):
                            predictions[idx] = translation.strip() if translation.strip() else f"[解码为空] {source_texts[idx]}"
                        
                        del translated_tokens
                        del inputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
        except Exception as e:
            if is_main_process():
                logger.error(f"批量推理失败: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 处理未推理的项目
        for i, pred in enumerate(predictions):
            if pred is None:
                predictions[i] = f"[推理失败] {source_texts[i]}"
        
        return predictions

    def log_inference_examples(self, model, original_dev_data, step, num_examples=5):
        """记录推理示例到wandb"""
        # 检查是否为主进程和 wandb 是否已初始化
        if not self.accelerator.is_main_process:
            return
            
        if original_dev_data is None:
            return
        
        # 检查 wandb 是否已初始化
        try:
            import wandb
            if not wandb.run:
                log_warning("wandb 未初始化，跳过推理示例记录")
                return
        except Exception as e:
            log_warning(f"wandb 检查失败，跳过推理示例记录: {e}")
            return
            
        try:
            # 随机选择示例
            data_size = len(original_dev_data)
            indices = random.sample(range(data_size), min(num_examples, data_size))
            
            sources = []
            references = []
            target_langs = []
            
            for idx in indices:
                example = original_dev_data[idx]
                sources.append(example['source_text'])
                references.append(example['target_text'])
                target_langs.append(example['target_lang'])
            
            # 批量推理
            predictions = self.inference_batch(model, sources, target_langs, batch_size=16)
            
            # 准备wandb表格数据
            for i, (source, target_lang, reference, prediction) in enumerate(zip(sources, target_langs, references, predictions)):
                examples_data.append([
                    step,
                    i + 1,
                    source,
                    target_lang,
                    reference,
                    prediction
                ])
            
            table = wandb.Table(
                columns=["评估步数", "示例编号", "中文原文", "目标语言", "标准答案", "预测结果"],
                data=examples_data
            )
            
            wandb.log({
                f"推理示例": table
            })
        except Exception as e:
            log_warning(f"记录推理示例失败: {e}")

class CustomSeq2SeqTrainer(Trainer):
    """自定义的序列到序列训练器"""
    
    def __init__(self, *args, evaluator=None, original_dev_data=None, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.original_dev_data = original_dev_data
        self.config = config
        self.best_metric = float('-inf')
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """自定义评估函数"""
        # 调用父类的评估获得损失
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 如果有评估器，计算翻译指标
        if self.evaluator is not None and self.original_dev_data is not None:
            bleu_score, comet_score, final_score = self.evaluator.evaluate_model(
                self.model, 
                self.original_dev_data, 
                num_samples=300
            )
            
            # 只在主进程中记录推理示例
            if is_main_process():
                self.evaluator.log_inference_examples(
                    self.model,
                    self.original_dev_data,
                    self.state.global_step,
                    num_examples=self.config.num_inference_examples if self.config else 10
                )
            
            # 添加总体翻译指标到评估结果
            eval_results.update({
                f"{metric_key_prefix}_bleu2": bleu_score,
                f"{metric_key_prefix}_comet": comet_score,
                f"{metric_key_prefix}_final": final_score
            })
            
            # 手动将总体评估结果记录到 wandb（确保指标被正确报告）
            if is_main_process():
                try:
                    import wandb
                    if wandb.run:
                        # 记录总体评估指标到 wandb（这些指标已经在evaluate_model中记录了分语言指标）
                        wandb_metrics = {
                            f"{metric_key_prefix}/overall_bleu2": bleu_score,
                            f"{metric_key_prefix}/overall_comet": comet_score, 
                            f"{metric_key_prefix}/overall_final": final_score,
                            f"{metric_key_prefix}/loss": eval_results.get(f"{metric_key_prefix}_loss", 0.0)
                        }
                        wandb.log(wandb_metrics)
                        log_info(f"已将总体评估指标记录到 wandb: {wandb_metrics}")
                except Exception as e:
                    log_warning(f"记录评估指标到 wandb 失败: {e}")
            
            # 检查是否为最佳模型
            if final_score > self.best_metric:
                self.best_metric = final_score
                log_info(f"新的最佳分数: {self.best_metric:.4f}")
        
        return eval_results

def train(config):
    """使用 Hugging Face Trainer 进行训练的主函数"""
    
    # 设置全局随机种子
    set_global_seed(config.seed)
    
    log_info(f"设置全局随机种子为: {config.seed}")
    log_info(f"训练模式: {'LoRA训练' if config.use_lora else '全量训练'}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 准备数据
    train_cache_path, dev_cache_path = get_processed_data_cache_path(config)
    
    if os.path.exists(train_cache_path) and os.path.exists(dev_cache_path):
        log_info("发现已处理的数据缓存，直接加载...")
        from datasets import load_from_disk
        train_dataset = load_from_disk(train_cache_path)
        dev_dataset = load_from_disk(dev_cache_path)
        
        model, tokenizers = prepare_model_and_tokenizer(config)
        
        log_info("重新加载原始开发集数据...")
        _, original_dev_dataset = prepare_data(config)
        
    else:
        log_info("未发现缓存，开始处理数据...")
        
        train_dataset, dev_dataset = prepare_data(config)
        original_dev_dataset = dev_dataset
        model, tokenizers = prepare_model_and_tokenizer(config)
        
        log_info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(dev_dataset)}")
        
        # 进行tokenization
        log_info("开始tokenization...")
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizers, config),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        dev_dataset = dev_dataset.map(
            lambda x: tokenize_function(x, tokenizers, config),
            batched=True,
            remove_columns=dev_dataset.column_names,
            desc="Tokenizing dev dataset"
        )
        
        # 保存处理好的数据
        log_info("保存处理好的数据到缓存...")
        train_dataset.save_to_disk(train_cache_path)
        dev_dataset.save_to_disk(dev_cache_path)
    
    # 创建数据整理器 - 根据训练语言选择合适的tokenizer
    if config.target_language == "all":
        collator_tokenizer = tokenizers['eng']  # 默认使用英语tokenizer
    else:
        # 单语言训练时使用对应的tokenizer
        lang_to_tokenizer_key = {
            "English": "eng", "英语": "eng",
            "Malay": "zsm", "马来语": "zsm",
            "Thai": "tha", "泰语": "tha"
        }
        tokenizer_key = lang_to_tokenizer_key.get(config.target_language, "eng")
        collator_tokenizer = tokenizers[tokenizer_key]
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=collator_tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 计算训练步数和评估步数
    num_train_epochs = config.num_epochs
    total_train_samples = len(train_dataset)
    per_device_train_batch_size = config.batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps
    
    # 使用固定的进程数量配置
    num_processes = 4
    
    # 计算每个epoch的实际训练步数
    # 考虑gradient accumulation和分布式训练
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_processes
    steps_per_epoch = total_train_samples // effective_batch_size
    
    # 如果有余数，需要额外一步
    if total_train_samples % effective_batch_size != 0:
        steps_per_epoch += 1
    
    # 计算总训练步数
    total_training_steps = steps_per_epoch * num_train_epochs
    # 直接使用配置中的评估步数
    eval_steps = config.eval_steps
    
    log_info(f"训练配置信息:")
    log_info(f"  - 总样本数: {total_train_samples}")
    log_info(f"  - 每设备batch size: {per_device_train_batch_size}")
    log_info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    log_info(f"  - 进程数量: {num_processes}")
    log_info(f"  - 有效batch size: {effective_batch_size}")
    log_info(f"  - 每epoch步数: {steps_per_epoch}")
    log_info(f"  - 总训练步数: {total_training_steps}")
    log_info(f"  - 评估频率: 每 {eval_steps} 步 (约{total_training_steps/eval_steps:.1f}次总评估)")
    
    # 设置wandb环境变量（让Trainer自动使用正确的project）
    if is_main_process():
        os.environ["WANDB_PROJECT"] = config.wandb_project
        log_info(f"设置wandb项目: {config.wandb_project}, 运行名称: {config.wandb_name}")
    
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=num_train_epochs,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,  # 使用配置的学习率调度类型
        logging_steps=1,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_final",
        greater_is_better=True,
        bf16=True,  # 使用 bfloat16 混合精度
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=config.wandb_name,
        seed=config.seed,
        data_seed=config.seed,
        disable_tqdm=False,
        prediction_loss_only=False,
        ddp_find_unused_parameters=False
    )
    
    # 初始化评估器 - 创建一个正确的评估器用于Trainer
    class TrainerAccelerator:
        @property
        def is_main_process(self):
            # 在 Trainer 环境中正确检查主进程
            return is_main_process()
    
    trainer_accelerator = TrainerAccelerator()
    evaluator = TranslationEvaluator(config, tokenizers, trainer_accelerator)
    
    # 创建自定义训练器
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        evaluator=evaluator,
        original_dev_data=original_dev_dataset,
        config=config,
    )
    
    log_info("开始训练...")
    log_info(f"总训练轮数: {num_train_epochs}")
    log_info(f"每{eval_steps}步进行一次评估")
    log_info(f"每个设备的batch size: {per_device_train_batch_size}")
    
    # 开始训练
    train_result = trainer.train()
    
    # 保存最终模型
    log_info("训练完成，保存最终模型...")
    final_model_dir = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    
    # 保存tokenizers
    for lang_key, tokenizer in tokenizers.items():
        tokenizer_dir = os.path.join(final_model_dir, f"tokenizer_{lang_key}")
        tokenizer.save_pretrained(tokenizer_dir)
        log_info(f"Tokenizer {lang_key} 保存完成")
    
    log_info(f"最终模型保存至: {final_model_dir}")
    log_info("训练完成！")

def test_inference(config):
    """测试推理功能，不进行训练"""
    log_info("="*50)
    log_info("开始推理测试模式")
    log_info("="*50)
    
    # 设置随机种子
    set_global_seed(config.seed)
    
    # 加载模型和tokenizer
    log_info("加载模型和tokenizer...")
    model, tokenizers = prepare_model_and_tokenizer(config)
    
    # 加载开发集数据
    log_info("加载开发集数据...")
    _, dev_dataset = prepare_data(config)
    log_info(f"开发集数据大小: {len(dev_dataset)}")
    
    # 创建一个简化的推理器
    class SimpleInferencer:
        def __init__(self, model, tokenizers, config):
            self.model = model
            self.tokenizers = tokenizers
            self.config = config
            
        def inference_single(self, source_text, target_lang):
            """单个文本推理 - 使用预初始化的多个tokenizer"""
            try:
                # 语言映射
                lang_to_tokenizer_key = {
                    "English": "eng",
                    "Malay": "zsm", 
                    "Thai": "tha",
                    "英语": "eng",
                    "马来语": "zsm",
                    "泰语": "tha"
                }
                
                # 选择对应的tokenizer
                tokenizer_key = lang_to_tokenizer_key.get(target_lang, "Thai")
                selected_tokenizer = self.tokenizers[tokenizer_key]
                target_lang_code = selected_tokenizer.tgt_lang
                
                # 编码输入文本
                inputs = selected_tokenizer(source_text, return_tensors="pt").to(self.model.device)
                
                with torch.inference_mode():
                    # 生成翻译 - 参考infer.py的方式
                    generate_kwargs = {
                        "max_length": 128,
                        "do_sample": False,
                        "num_beams": 4,
                        "early_stopping": True
                    }
                    
                    generate_kwargs["forced_bos_token_id"] = selected_tokenizer.convert_tokens_to_ids(target_lang_code)
                    
                    translated_tokens = self.model.generate(**inputs, **generate_kwargs)
                
                # 解码翻译结果
                translated_text = selected_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                # 清理缓存
                del translated_tokens
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 简单的结果检查
                if not translated_text.strip():
                    return f"[解码为空] {source_text}"
                else:
                    return translated_text.strip()
                
            except Exception as e:
                if is_main_process():
                    logger.error(f"推理失败: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return f"[推理失败: {str(e)}]"
    
    # 创建推理器
    inferencer = SimpleInferencer(model, tokenizers, config)
    
    # 随机选择5个示例进行推理测试
    log_info("\n" + "="*50)
    log_info("开始推理测试 (5个示例):")
    log_info("="*50)
    
    import random
    random.seed(config.seed)
    indices = random.sample(range(len(dev_dataset)), min(5, len(dev_dataset)))
    
    for i, idx in enumerate(indices):
        example = dev_dataset[idx]
        
        source_text = example['source_text']
        target_text = example['target_text']
        target_lang = example['target_lang']
        
        log_info(f"\n=== 测试示例 {i+1} ===")
        log_info(f"中文原文: {source_text}")
        log_info(f"目标语言: {target_lang}")
        log_info(f"标准答案: {target_text}")
        
        # 进行推理
        predicted_text = inferencer.inference_single(source_text, target_lang)
        
        log_info(f"预测结果: {predicted_text}")
        log_info("-" * 50)
    
    log_info("\n推理测试完成!")

if __name__ == "__main__":
    config = TrainingConfig()
    
    # 可以通过命令行参数设置测试模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        config.test_only = True
        log_info("启用测试模式")
    
    if config.test_only:
        test_inference(config)
    else:
        train(config) 