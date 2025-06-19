#!/bin/bash

# accelerate多卡训练启动脚本

# 检查accelerate配置
echo "检查accelerate配置..."

# 启动多卡训练
echo "启动NLLB多卡训练..."
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 29500 \
    train_nllb.py

echo "训练完成！" []