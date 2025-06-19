#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 使用accelerate启动训练，从配置文件读取设置
accelerate launch \
    --config_file ../train/accelerate_config.yaml \
    ../train/train_gemmax2.py

echo "训练完成！" 