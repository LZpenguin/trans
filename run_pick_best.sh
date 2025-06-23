#!/bin/bash

# 多模型翻译结果选择脚本示例
# 用法示例：从多个模型的翻译结果中选择最佳翻译

echo "多模型翻译结果选择器"
echo "===================="

# 设置输入文件路径
INPUT_FILES=(
    "output/gemini/dev_gp_en_30shot.csv"
    "output/openai/dev_4o_30shot.csv"
    "output/deepseek/dev_p2_t0_rag_30shot.csv"
    "output/deepseek/dev_p2_t0_rag_context_30shot.csv"
)

# 设置输出路径
OUTPUT_FILE="output/best.csv"

# 检查输入文件是否存在
echo "检查输入文件..."
FILES_EXIST=0
for file in "${INPUT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file 存在"
        FILES_EXIST=1
    else
        echo "✗ $file 不存在"
    fi
done

if [ $FILES_EXIST -eq 0 ]; then
    echo "错误：没有找到任何输入文件"
    echo "请检查以下文件是否存在："
    for file in "${INPUT_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

# 运行选择器
echo ""
echo "开始选择最佳翻译..."
echo "输出文件：$OUTPUT_FILE"
echo ""

# 基础用法
python pick_best.py \
    --input-files "${INPUT_FILES[@]}" \
    --output "$OUTPUT_FILE" \

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "选择完成！"
    echo "结果文件：$OUTPUT_FILE"
    echo "统计文件：${OUTPUT_FILE//.csv/_stats.json}"
else
    echo ""
    echo "选择失败！请检查错误信息"
fi 