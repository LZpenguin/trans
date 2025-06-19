#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单语言训练示例脚本

这个脚本展示了如何使用 train_nllb.py 来训练单一目标语言的翻译模型。
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_nllb import TrainingConfig, train, test_inference

def train_english_only():
    """训练只针对英语的翻译模型"""
    print("=" * 60)
    print("训练中文->英语翻译模型")
    print("=" * 60)
    
    config = TrainingConfig()
    
    # 指定只训练英语
    config.target_language = "English"  # 或者使用 "英语"
    
    # 调整训练参数
    config.batch_size = 32
    config.learning_rate = 2e-5
    config.num_epochs = 5
    config.training_idx = 201  # 避免与其他实验冲突
    
    # 根据目标语言更新相关配置
    config.lang_suffix = f"-{config.target_language}" if config.target_language != "all" else ""
    config.output_dir = os.path.join(config.base_dir, f"models/{config.training_method}/{config.model_path.split('/')[-1]}-{config.training_method}-{config.training_mode}-{config.training_idx}{config.lang_suffix}")
    config.wandb_name = f"nllb-translation-{config.training_method}-{config.training_mode}-{config.training_idx}{config.lang_suffix}"
    
    print(f"目标语言: {config.target_language}")
    print(f"输出目录: {config.output_dir}")
    print(f"Wandb运行名称: {config.wandb_name}")
    
    # 开始训练
    train(config)

def train_thai_only():
    """训练只针对泰语的翻译模型"""
    print("=" * 60)
    print("训练中文->泰语翻译模型")
    print("=" * 60)
    
    config = TrainingConfig()
    
    # 指定只训练泰语
    config.target_language = "Thai"  # 或者使用 "泰语"
    
    # 调整训练参数
    config.batch_size = 16
    config.learning_rate = 2e-5
    config.num_epochs = 5
    config.training_idx = 202  # 避免与其他实验冲突
    
    # 根据目标语言更新相关配置
    config.lang_suffix = f"-{config.target_language}" if config.target_language != "all" else ""
    config.output_dir = os.path.join(config.base_dir, f"models/{config.training_method}/{config.model_path.split('/')[-1]}-{config.training_method}-{config.training_mode}-{config.training_idx}{config.lang_suffix}")
    config.wandb_name = f"nllb-translation-{config.training_method}-{config.training_mode}-{config.training_idx}{config.lang_suffix}"
    
    print(f"目标语言: {config.target_language}")
    print(f"输出目录: {config.output_dir}")
    print(f"Wandb运行名称: {config.wandb_name}")
    
    # 开始训练
    train(config)

def train_malay_only():
    """训练只针对马来语的翻译模型"""
    print("=" * 60)
    print("训练中文->马来语翻译模型")
    print("=" * 60)
    
    config = TrainingConfig()
    
    # 指定只训练马来语
    config.target_language = "Malay"  # 或者使用 "马来语"
    
    # 调整训练参数
    config.batch_size = 16
    config.learning_rate = 2e-5
    config.num_epochs = 5
    config.training_idx = 203  # 避免与其他实验冲突
    
    # 根据目标语言更新相关配置
    config.lang_suffix = f"-{config.target_language}" if config.target_language != "all" else ""
    config.output_dir = os.path.join(config.base_dir, f"models/{config.training_method}/{config.model_path.split('/')[-1]}-{config.training_method}-{config.training_mode}-{config.training_idx}{config.lang_suffix}")
    config.wandb_name = f"nllb-translation-{config.training_method}-{config.training_mode}-{config.training_idx}{config.lang_suffix}"
    
    print(f"目标语言: {config.target_language}")
    print(f"输出目录: {config.output_dir}")
    print(f"Wandb运行名称: {config.wandb_name}")
    
    # 开始训练
    train(config)

def test_single_language():
    """测试单语言模型的推理"""
    print("=" * 60)
    print("测试单语言模型推理")
    print("=" * 60)
    
    config = TrainingConfig()
    config.target_language = "Thai"  # 指定测试泰语模型
    config.test_only = True
    
    test_inference(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="单语言训练示例")
    parser.add_argument("--lang", choices=["english", "thai", "malay", "test"], 
                       help="选择训练的语言或测试模式")
    
    args = parser.parse_args()
    
    if args.lang == "english":
        train_english_only()
    elif args.lang == "thai":
        train_thai_only()
    elif args.lang == "malay":
        train_malay_only()
    elif args.lang == "test":
        test_single_language()
    else:
        print("使用示例:")
        print("python train_single_lang_example.py --lang english   # 训练英语")
        print("python train_single_lang_example.py --lang thai      # 训练泰语")  
        print("python train_single_lang_example.py --lang malay     # 训练马来语")
        print("python train_single_lang_example.py --lang test      # 测试推理")
        print()
        print("或者直接修改 train_nllb.py 中的 TrainingConfig.target_language 参数:")
        print("  - 设置为 'English' 或 '英语' 只训练英语")
        print("  - 设置为 'Thai' 或 '泰语' 只训练泰语")
        print("  - 设置为 'Malay' 或 '马来语' 只训练马来语")
        print("  - 设置为 'all' 训练所有语言（默认）") 