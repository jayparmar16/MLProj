"""Evaluate the trained BIO+CRF model under the SAME span-F1 metric used in
train_biaffine.py — i.e. exact-match (start, end, class) tuples, micro-averaged.

This gives an apples-to-apples baseline against the biaffine span head's number.

Run:
  python evaluate_bio_as_spans.py
"""
import os
import torch
from tqdm import tqdm

from dataset import get_dataloaders
from model import SkillExtractor
from train_biaffine import gold_spans_from_bio, span_f1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_bio(model, loader, device, use_crf=True):
    """Return a list (per-example) of predicted BIO label tensors of shape (L,)."""
    model.eval()
    preds_per_example = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            if use_crf:
                # CRF Viterbi decoding — already respects masking
                preds = model.crf.decode(logits.float(), attention_mask.float())
            else:
                preds = logits.argmax(dim=-1)
            # Zero out padding positions so they don't generate spurious spans
            preds = preds.masked_fill(attention_mask == 0, 0)
            for i in range(preds.size(0)):
                preds_per_example.append(preds[i].cpu())
    return preds_per_example


def stack_to_padded(tensor_list, pad_to):
    """Stack variable-length 1-D tensors into (N, pad_to). All inputs are already pad_to."""
    return torch.stack(tensor_list, dim=0)


def main(
    ckpt="./checkpoints/best_model.pt",
    batch_size=32,
    max_length=128,
    num_layers=6,
    d_ff=768,
    dropout=0.2,
    use_crf=True,
):
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    model = SkillExtractor(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=5,
        max_len=max_length,
        dropout=dropout,
        use_crf=use_crf,
    ).to(device)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"BIO checkpoint not found at {ckpt}. Run train.py first.")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded BIO+CRF model from {ckpt}")

    # ---- Test set ----
    print("\nEvaluating on TEST set...")
    pred_bio = predict_bio(model, test_loader, device, use_crf=use_crf)
    # Collect gold labels (already padded)
    gold_bio = []
    for batch in test_loader:
        for i in range(batch["labels"].size(0)):
            gold_bio.append(batch["labels"][i])

    # Convert both sides to spans via the same routine
    pred_padded = stack_to_padded(pred_bio, max_length)
    gold_padded = stack_to_padded(gold_bio, max_length)

    pred_spans = gold_spans_from_bio(pred_padded)   # name is misleading: it just walks BIO
    gold_spans = gold_spans_from_bio(gold_padded)

    p, r, f = span_f1(pred_spans, gold_spans)
    print("\n" + "=" * 60)
    print("  BIO+CRF predictions, scored as SPANS (exact match)")
    print("=" * 60)
    print(f"  Span Precision: {p:.4f}")
    print(f"  Span Recall:    {r:.4f}")
    print(f"  Span F1:        {f:.4f}")
    print("=" * 60)

    # Per-class span F1
    for cls_id, cls_name in [(1, "Skill"), (2, "Knowledge")]:
        pred_c = [[s for s in spans if s[2] == cls_id] for spans in pred_spans]
        gold_c = [[s for s in spans if s[2] == cls_id] for spans in gold_spans]
        pc, rc, fc = span_f1(pred_c, gold_c)
        print(f"  {cls_name:>10} span F1: {fc:.4f} (P {pc:.4f} / R {rc:.4f})")


if __name__ == "__main__":
    main()
