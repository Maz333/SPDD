from transformers import AutoModelForTokenClassification

def build_model(num_labels, model_name="bert-base-chinese"):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
