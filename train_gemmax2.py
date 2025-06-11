import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from datasets import Dataset, concatenate_datasets
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import warnings
from tqdm import tqdm
import logging
import wandb
import numpy as np
import random
from dataset import TranslationDataset


ft_idx = 101

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings('ignore')

# 设置随机种子函数
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# 配置参数
class TrainingConfig:
    model_path = "/home/zbtrs/lz/trans/models/GemmaX2-28-2B-v0.1"
    train_parquet_files = [
        "/home/zbtrs/lz/trans/v2/opus-100/en-zh/train-00000-of-00001.parquet",
        "/home/zbtrs/lz/trans/v2/opus-100/en-ms/train-00000-of-00001.parquet",
        "/home/zbtrs/lz/trans/v2/opus-100/en-th/train-00000-of-00001.parquet"
    ]
    dev_file = "/home/zbtrs/lz/trans/data/text_data/dev.csv"
    output_dir = f"/home/zbtrs/lz/trans/models/pt/GemmaX2-28-2B-v0.1-pt-{ft_idx}"
    
    # LoRA配置
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    
    # 训练配置
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 1
    max_length = 128
    gradient_accumulation_steps = 1
    eval_save_num = 50
    
    # wandb配置
    wandb_project = "gemmax2-translation-pt"
    wandb_name = f"gemmax2-translation-pt-{ft_idx}"
    
    # 添加随机种子
    seed = 42

def prepare_data(config):
    # 初始化数据集加载器
    dataset_loader = TranslationDataset(data_type="pt", double=True)
    
    # 加载预训练数据
    train_datasets = []
    for parquet_file in config.train_parquet_files:
        logger.info(f"加载预训练数据: {parquet_file}")
        dataset = dataset_loader.load_data(parquet_file)
        train_datasets.append(dataset)
    
    # 合并所有预训练数据集
    train_dataset = concatenate_datasets(train_datasets)
    logger.info(f"合并后的预训练数据集大小: {len(train_dataset)}")
    
    # 加载开发集数据
    dev_dataset_loader = TranslationDataset(data_type="ft", double=False)
    dev_dataset = dev_dataset_loader.load_data(config.dev_file)
    logger.info(f"开发集数据大小: {len(dev_dataset)}")
    
    return train_dataset, dev_dataset

def prepare_model_and_tokenizer(config):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map="auto"
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # 获取PEFT模型
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

lang_map = {
    "马来语": "malay",
    "泰语": "thai",
    "英语": "english"
}

def tokenize_function(examples, tokenizer, config):
    # 所有数据集现在都有统一的prompt字段
    prompts = examples['prompt']
    
    # 编码输入
    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors='pt'
    )
    
    labels = inputs['input_ids'].clone()
    
    # 获取冒号的token_id
    colon_token_id = tokenizer.encode(':',add_special_tokens=False)[0]
    
    # 将input_ids转换为numpy数组进行处理
    input_ids_np = inputs['input_ids'].numpy()
    
    # 找到所有冒号的位置
    colon_mask = (input_ids_np == colon_token_id)
    
    # 对每个序列处理
    for i in range(len(input_ids_np)):
        # 找到当前序列中最后一个冒号的位置
        colon_positions = np.where(colon_mask[i])[0]
        if len(colon_positions) > 0:
            last_colon_pos = colon_positions[-1]
            # 将最后一个冒号之前的所有token标记为-100
            labels[i, :last_colon_pos+1] = -100
    
    inputs['labels'] = labels
    labels[labels == tokenizer.pad_token_id] = -100
    
    return inputs

def train(config):
    # 设置全局随机种子
    set_global_seed(config.seed)
    logger.info(f"设置全局随机种子为: {config.seed}")
    
    # 初始化wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_name,
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.num_epochs,
            "model": config.model_path,
        }
    )
    
    # 准备数据
    train_dataset, dev_dataset = prepare_data(config)
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(dev_dataset)}")
    
    # 准备模型和tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)
    logger.info("模型和tokenizer加载完成")
    
    # 数据预处理
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=train_dataset.column_names,
        cache_file_name=f"{config.output_dir}/train_cache.arrow"
    )
    
    dev_dataset = dev_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=dev_dataset.column_names,
        cache_file_name=f"{config.output_dir}/dev_cache.arrow"
    )
    
    # 打印多个样本示例
    logger.info("打印前5个训练样本的示例：")
    for i in range(1):
        print(f"\n=== 样本 {i+1} ===")
        sample = train_dataset[i]
        print(f"\ninput_ids:\n{sample['input_ids']}")
        print(f"\nlabels:\n{sample['labels']}")
        print("\n解码后的文本：")
        print(f"输入文本：{tokenizer.decode(sample['input_ids'])}")
        print(f"标签文本：{tokenizer.decode([id for id in sample['labels'] if id != -100])}")
        print("="*50)

    # 计算每5%进度对应的步数
    total_steps = (len(train_dataset) * config.num_epochs) // (config.batch_size * config.gradient_accumulation_steps)
    # eval_steps = max(1, total_steps // config.eval_save_num)
    eval_steps = 1000
    logger.info(f"总训练步数: {total_steps}, 每{eval_steps}步进行一次评估")
    
    # 设置训练参数[/home/zbtrs/lz/trans/models]
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=1,
        save_strategy="steps",
        save_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=config.eval_save_num,  # 只保存最好的3个检查点
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_safetensors=True,  # 使用safetensors格式保存
        overwrite_output_dir=True,
        # 添加seed参数
        seed=config.seed,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    
    # 检查是否存在检查点
    last_checkpoint = None
    if os.path.exists(config.output_dir):
        checkpoints = [f for f in os.listdir(config.output_dir) if f.startswith("checkpoint")]
        if len(checkpoints) > 0:
            # 按照检查点编号排序
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            last_checkpoint = os.path.join(config.output_dir, checkpoints[-1])
            logger.info(f"发现最新的检查点: {last_checkpoint}")
    
    # 开始训练，如果存在检查点则从检查点继续训练
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"模型保存至: {config.output_dir}")
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    config = TrainingConfig()
    train(config) 