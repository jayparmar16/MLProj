import torch
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from dataset import ID_TO_LABEL
import time

def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            if criterion is not None:
                # Reshape for loss calculation
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.classifier.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = criterion(active_logits, active_labels)
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            # Move to CPU for metrics
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            # Mask out padding and special tokens (-100) using boolean array indexing
            mask = labels != -100
            all_preds.extend(preds[mask].tolist())
            all_labels.extend(labels[mask].tolist())

    inference_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader) if criterion is not None else 0

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return {
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": inference_time,
        "all_preds": all_preds,
        "all_labels": all_labels
    }

def print_detailed_metrics(eval_results):
    labels = eval_results["all_labels"]
    preds = eval_results["all_preds"]

    target_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=target_names, labels=list(range(len(ID_TO_LABEL))), zero_division=0))
    print(f"Total Inference Time: {eval_results['inference_time']:.2f}s")
