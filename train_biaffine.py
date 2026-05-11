"""Train the biaffine span head as an alternative to BIO+CRF.

The span head scores every (start, end) token pair as one of {None, Tech, Knowledge}.
Decoding is greedy non-overlapping over the span score grid (see model.decode_spans).

Run alongside the BIO+CRF train.py for an apples-to-apples F1 comparison.
KD/teacher integration is left out of v1 — once the span-head baseline is logged
in SCORES.md, hidden-state KD can be re-added the same way as in train.py.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataset import get_dataloaders, ID_TO_LABEL
from model import SpanSkillExtractor, bio_to_span_labels, decode_spans


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gold_spans_from_bio(bio_labels):
    """Extract (start, end, class) spans from a (B, L) BIO tensor. class: 1=Tech, 2=Knowledge."""
    B, L = bio_labels.shape
    out = []
    for b in range(B):
        spans, i = [], 0
        while i < L:
            tag = int(bio_labels[b, i])
            if tag in (1, 3):
                cls = 1 if tag == 1 else 2
                inside = tag + 1
                end = i
                while end + 1 < L and int(bio_labels[b, end + 1]) == inside:
                    end += 1
                spans.append((i, end, cls))
                i = end + 1
            else:
                i += 1
        out.append(spans)
    return out


def span_f1(pred_spans, gold_spans):
    """Micro F1 over (start, end, class) tuples — exact-match span scoring."""
    tp = fp = fn = 0
    for preds, golds in zip(pred_spans, gold_spans):
        pred_set, gold_set = set(preds), set(golds)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate_span_model(model, loader, device, max_span_len=20):
    model.eval()
    all_pred, all_gold = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]  # keep on CPU for span extraction

            scores = model(input_ids, attention_mask)  # (B, L, L, C)
            preds = decode_spans(scores, attention_mask, max_span_len=max_span_len)
            golds = gold_spans_from_bio(labels)
            all_pred.extend(preds)
            all_gold.extend(golds)
    return span_f1(all_pred, all_gold)


def train(
    epochs=50,
    batch_size=32,
    lr=5e-4,
    max_length=128,
    save_dir="./checkpoints",
    warmup_ratio=0.1,
    accum_steps=2,
    patience=10,
    num_layers=6,
    d_ff=768,
    dropout=0.2,
    d_span_proj=64,
    max_span_len=20,
    none_weight=0.3,  # downweight the dominant "None" class
):
    device = get_device()
    print("=" * 60)
    print("  SKILL EXTRACTOR — BIAFFINE SPAN HEAD")
    print("=" * 60)
    print(f"Device: {device} | Layers: {num_layers} | d_ff: {d_ff} | "
          f"d_span_proj: {d_span_proj} | max_span_len: {max_span_len}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    model = SpanSkillExtractor(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=num_layers,
        d_ff=d_ff,
        num_span_classes=3,
        max_len=max_length,
        dropout=dropout,
        d_span_proj=d_span_proj,
    ).to(device)
    print(f"Params: {model.get_num_parameters():,} (budget 6M)")

    # Class weights: heavy down-weight on the "None" class — the L*L grid is
    # >99% None. Without this, the model trivially predicts everything None.
    class_weights = torch.tensor([none_weight, 1.0, 1.0], device=device)
    ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader) // accum_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_labels = batch["labels"].to(device)
            span_labels = bio_to_span_labels(bio_labels, max_span_len=max_span_len)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                scores = model(input_ids, attention_mask)  # (B, L, L, C)
                loss = ce(scores.reshape(-1, scores.size(-1)), span_labels.reshape(-1))
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            bar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}",
                             "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        p, r, f = evaluate_span_model(model, val_loader, device, max_span_len)
        print(f"  Epoch {epoch+1:>2}/{epochs} | TrainLoss {total_loss/len(train_loader):.4f} "
              f"| ValP {p:.4f} | ValR {r:.4f} | ValF1 {f:.4f}")

        if f > best_f1:
            best_f1 = f
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{save_dir}/best_biaffine_model.pt")
            print(f"  >>> New best span F1: {best_f1:.4f} — saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    print("\n" + "=" * 60)
    print("  FINAL TEST EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load(f"{save_dir}/best_biaffine_model.pt", map_location=device))
    p, r, f = evaluate_span_model(model, test_loader, device, max_span_len)
    print(f"Test span P/R/F1: {p:.4f} / {r:.4f} / {f:.4f}")
    print(f"Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
