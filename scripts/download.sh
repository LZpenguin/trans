# 下载这个链接 git clone https://hf-mirror.com/datasets/Helsinki-NLP/opus-100
REPO_URL="https://hf-mirror.com/datasets/Helsinki-NLP/opus-100"

mkdir -p ../data
mkdir -p ../models

cd ../data

git clone $REPO_URL

cd ../models

git clone git clone https://hf-mirror.com/FireRedTeam/FireRedASR-AED-L

git clone https://hf-mirror.com/ModelSpace/GemmaX2-28-9B-v0.1