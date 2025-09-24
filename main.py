import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# print("CUDA available:", torch.cuda.is_available())
# print("Device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# import sys, transformers
# print("Python:", sys.executable)
# print("Transformers version:", transformers.__version__)
# print("TrainingArguments real location:", transformers.training_args.__file__)


from datasets import load_dataset
from transformers import AutoTokenizer
from dataset.conll_dataset import load_conll_dataset
from model.ner_model import build_model
from train.trainer import build_trainer


def main():
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")

    # 2. 加载数据
    dataset = load_conll_dataset(
        "data/weiboNER_2nd_conll.train",
        "data/weiboNER_2nd_conll.dev",
        "data/weiboNER_2nd_conll.test",
        tokenizer
    )

    # 3. 构建模型
    model = build_model(
        num_labels=len(dataset.label_list),
        model_name="bert-base-chinese"
    )
    model.config.label2id = dataset.label2id
    model.config.id2label = dataset.id2label
    model.to(device)

    # 4. 构建 Trainer
    trainer = build_trainer(model, dataset, tokenizer)

    # 5. 训练
    trainer.train()

    # 6. 评估
    print("Eval dataset keys:", dataset["test"].column_names)
    # metrics = trainer.evaluate()
    # print(metrics)
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"])
    print("Test metrics:", test_metrics)
    trainer.save_model("./best_model")



if __name__ == "__main__":
    main()
