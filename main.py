from datasets import load_dataset
from transformers import AutoTokenizer
from dataset.conll_dataset import load_conll_dataset
from model.ner_model import build_model
from train.trainer import build_trainer

def main():
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

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

    # 4. 构建 Trainer
    trainer = build_trainer(model, dataset, tokenizer)

    # 5. 训练
    trainer.train()

    # 6. 评估
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()