import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from transformers import BertForTokenClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from dataset import get_dataloaders
from evaluate import evaluate_model, print_detailed_metrics

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_teacher(
    epochs=15,
    batch_size=32,
    lr=3e-5,
    max_length=128,
    save_dir="./checkpoints",
    weight_decay=0.1,
    label_smoothing=0.05,
    dropout=0.2,
    patience=4,
):
    device = get_device()
    print(f"{'='*60}")
    print(f"  TEACHER MODEL (BERT-base) TRAINING — regularized")
    print(f"{'='*60}")
    print(f"Device:           {device}")
    print(f"Epochs:           {epochs} (early stop patience={patience})")
    print(f"Batch size:       {batch_size}")
    print(f"Learning rate:    {lr}  (warmup+cosine)")
    print(f"Weight decay:     {weight_decay}")
    print(f"Label smoothing:  {label_smoothing}")
    print(f"Dropout (h/a/cls): {dropout}")
    print(f"{'='*60}\n")

    os.makedirs(save_dir, exist_ok=True)

    print("Loading datasets...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    print("\nInitializing BERT-base Teacher Model...")
    # Bump dropout on attention, hidden, and the classifier head — previous 5-epoch
    # run showed clear overfitting (train loss collapsing while val F1 plateaued).
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        classifier_dropout=dropout,
    ).to(device)

    # Manual loss so we can apply label smoothing (HF's `labels=` path uses CE w/o smoothing).
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = 0.0
    epochs_without_improvement = 0

    print("Starting Teacher training...\n")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # No `labels=` so we get raw logits, then apply label-smoothed CE manually.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        print(f"\nEvaluating Teacher on validation set...")
        val_results = evaluate_model(model, val_loader, device)
        print(f"  Epoch {epoch+1:>2}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_results['f1']:.4f}")

        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            epochs_without_improvement = 0
            print(f"  >>> New best F1: {best_f1:.4f} — saving teacher checkpoint")
            torch.save(model.state_dict(), f"{save_dir}/teacher_bert_best_model.pt")
        else:
            epochs_without_improvement += 1
            print(f"  --- No improvement for {epochs_without_improvement}/{patience} epochs")
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch+1} epochs.")
                break
            
    # Final evaluation
    print(f"\n{'='*60}")
    print("  FINAL EVALUATION ON TEST SET (TEACHER)")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(f"{save_dir}/teacher_bert_best_model.pt", map_location=device))
    
    # HuggingFace models return an object, so we need to slightly wrap our evaluate_model 
    # to handle outputs.logits if it's an HF model. We'll update evaluate.py briefly or just pass a wrapper.
    # Actually evaluate.py expects model(input, mask) to return logits. 
    # DistilBertForTokenClassification returns TokenClassifierOutput when labels are provided, 
    # but when labels are NOT provided, it returns (logits,).
    # Oh wait, evaluate.py just calls `logits = model(input_ids, attention_mask)`.
    # For HF model without labels, model(...) returns an object where outputs.logits is the tensor.
    # Let's adjust evaluate.py to handle this!

    print("\nBenchmarking Teacher inference on CPU...")
    model.to("cpu")
    cpu_test_results = evaluate_model(model, test_loader, torch.device("cpu"))
    print_detailed_metrics(cpu_test_results)
    print(f"Teacher Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_teacher()
