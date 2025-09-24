import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import EvalPrediction

def build_compute_metrics(id2label):
    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        # 去掉 padding
        true_predictions = [
            [id2label[pred] for (pred, lab) in zip(pred_row, label_row) if lab != -100]
            for pred_row, label_row in zip(preds, labels)
        ]
        true_labels = [
            [id2label[lab] for (pred, lab) in zip(pred_row, label_row) if lab != -100]
            for pred_row, label_row in zip(preds, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    return compute_metrics
