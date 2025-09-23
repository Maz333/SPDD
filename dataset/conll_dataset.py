import datasets

def read_conll_file(path):
    sentences = []
    tokens, tags = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])
        if tokens:  # 最后一条
            sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


def load_conll_dataset(train_path, dev_path, test_path, tokenizer, max_length=256):
    # 1. 读取三个文件
    train_data = read_conll_file(train_path)
    dev_data   = read_conll_file(dev_path)
    test_data  = read_conll_file(test_path)

    # 2. 构建 DatasetDict
    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(train_data),
        "validation": datasets.Dataset.from_list(dev_data),
        "test": datasets.Dataset.from_list(test_data),
    })

    # 3. 构建标签集
    all_tags = set()
    for row in dataset["train"]["ner_tags"]:
        all_tags.update(row)
    label_list = sorted(list(all_tags))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    # 4. 定义 tokenization + label 对齐
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label2id[label[word_id]])
                else:
                    # 对 subword 用 -100 忽略
                    label_ids.append(-100)
                previous_word_id = word_id
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 5. 转换成 tokenized dataset
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # 6. 附加元数据
    tokenized_dataset.label_list = label_list
    tokenized_dataset.label2id = label2id
    tokenized_dataset.id2label = id2label

    return tokenized_dataset
