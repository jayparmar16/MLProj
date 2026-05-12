"""Per-class span-F1 for the saved biaffine checkpoint.

Loads ./checkpoints/best_biaffine_model.pt, runs the test set, and reports
overall + per-class (Tech, Knowledge) span F1 — same exact-match metric as
evaluate_bio_as_spans.py and train_biaffine.py.

Run:
  python evaluate_biaffine_per_class.py
"""
import os
import torch
from tqdm import tqdm

from dataset import get_dataloaders
from model import SpanSkillExtractor, decode_spans
from train_biaffine import gold_spans_from_bio, span_f1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(
    ckpt="./checkpoints/best_biaffine_model.pt",
    batch_size=32,
    max_length=128,
    num_layers=6,
    d_ff=768,
    dropout=0.2,
    d_span_proj=64,
    max_span_len=20,
):
    device = get_device()
    print(f"Device: {device}")

    _, _, test_loader, tokenizer = get_dataloaders(
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

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Biaffine checkpoint not found at {ckpt}.")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded biaffine model from {ckpt}")

    model.eval()
    all_pred, all_gold = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            scores = model(input_ids, attention_mask)
            preds = decode_spans(scores, attention_mask, max_span_len=max_span_len)
            golds = gold_spans_from_bio(labels)
            all_pred.extend(preds)
            all_gold.extend(golds)

    p, r, f = span_f1(all_pred, all_gold)
    print("\n" + "=" * 60)
    print("  Biaffine span head — TEST set, exact-match span metric")
    print("=" * 60)
    print(f"  Overall span F1: {f:.4f} (P {p:.4f} / R {r:.4f})")
    for cls_id, cls_name in [(1, "Skill"), (2, "Knowledge")]:
        pred_c = [[s for s in spans if s[2] == cls_id] for spans in all_pred]
        gold_c = [[s for s in spans if s[2] == cls_id] for spans in all_gold]
        pc, rc, fc = span_f1(pred_c, gold_c)
        print(f"  {cls_name:>10} span F1: {fc:.4f} (P {pc:.4f} / R {rc:.4f})")


if __name__ == "__main__":
    main()
