from transformers import TrainingArguments, Trainer
from utils.metrics import compute_metrics

def build_trainer(model, tokenized_dataset, tokenizer, output_dir="./results"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        no_cuda=False,   # ✅ 确保允许用 CUDA
        fp16=True 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer
