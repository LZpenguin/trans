import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

torch.set_float32_matmul_precision('high')


base_model_id = "/home/zbtrs/lz/trans/models/GemmaX2-28-2B-v0.1"
adapter_model_id = "/home/zbtrs/lz/trans/models/GemmaX2-28-2B-v0.1-finetuned-9/checkpoint-9375"

# 首先加载基础模型和分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, adapter_model_id)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

lang = 'malay'
text = f"Translate this from Chinese to {lang}:\nChinese: 我爱机器翻译\n{lang}:"
inputs = tokenizer(text, return_tensors="pt")

# 将输入数据移动到GPU
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))