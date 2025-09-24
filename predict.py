import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 1. 加载保存好的模型和分词器
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# 2. 构建 NER pipeline
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

# 3. 输入新句子
sentence = "小明在北京大学学习人工智能，他的身份证和手机号是111."
results = ner(sentence)

print(results)
