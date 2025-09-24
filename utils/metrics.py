from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataset.conll_dataset import id2label

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    # 去掉 padding（label = -100 的部分）
    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }
