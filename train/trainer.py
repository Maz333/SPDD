from transformers import TrainingArguments, Trainer
from utils.metrics import build_compute_metrics

def build_trainer(model, tokenized_dataset, tokenizer, output_dir="./results"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,   # ✅ 自动加载最优模型
        metric_for_best_model="f1",    # ✅ 用 f1 作为判定标准
        greater_is_better=True,        # ✅ 越大越好
        save_total_limit=1, 
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        no_cuda=False,   # ✅ 确保允许用 CUDA
        fp16=True,
        prediction_loss_only=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenized_dataset.id2label)
    )
    return trainer
