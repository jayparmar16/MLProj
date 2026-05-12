"""Generate pseudo-labels from unlabeled job descriptions using the teacher model.

Inputs:
  - A text file with one job description per line (configurable path).
  - The trained DistilBERT teacher checkpoint.

Outputs:
  - A `.pt` file containing a list of dicts:
      {"input_ids": LongTensor(L), "attention_mask": LongTensor(L), "labels": LongTensor(L)}
    Tokens whose teacher confidence is below `min_confidence` are masked to -100 so
    the student is not forced to learn them. Sequences with too few confident
    label tokens are dropped entirely.

Usage:
  python pseudo_label.py --input data/unlabeled_jds.txt --output data/pseudo.pt

Then in train.py, set `pseudo_data_path="data/pseudo.pt"` to merge into training.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, BertForTokenClassification
from tqdm import tqdm


def load_teacher(ckpt_path, num_labels=5, device="cpu"):
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    ).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded teacher from {ckpt_path}")
    else:
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {ckpt_path}. "
            f"Train the teacher first via teacher_model.py."
        )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .txt file: one JD per line.")
    parser.add_argument("--output", default="data/pseudo.pt")
    parser.add_argument("--teacher", default="./checkpoints/teacher_bert_best_model.pt")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--min_confidence", type=float, default=0.9,
        help="Per-token softmax max prob. Tokens below this become -100 (ignored in loss).",
    )
    parser.add_argument(
        "--min_kept_fraction", type=float, default=0.5,
        help="Drop a sequence if fewer than this fraction of its real tokens survived filtering.",
    )
    parser.add_argument(
        "--min_skill_tokens", type=int, default=1,
        help="Drop a sequence if it has fewer than this many non-O confident tokens "
             "(prevents the dataset from being all-'O' pseudo-noise).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher = load_teacher(args.teacher, num_labels=5, device=device)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"Read {len(lines)} unlabeled descriptions from {args.input}.")

    out_records = []
    dropped_low_kept = 0
    dropped_no_skill = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(lines), args.batch_size), desc="Pseudo-labeling"):
            batch_text = lines[start:start + args.batch_size]
            enc = tokenizer(
                batch_text,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = teacher(input_ids, attention_mask=attention_mask).logits
            probs = logits.softmax(dim=-1)
            conf, preds = probs.max(dim=-1)  # (B, L)

            # Mask: confident AND a real (non-pad) token
            real = attention_mask.bool()
            confident = conf >= args.min_confidence
            keep = real & confident
            labels = preds.masked_fill(~keep, -100)

            # Per-sequence quality filters
            for i in range(input_ids.size(0)):
                real_count = real[i].sum().item()
                kept_count = keep[i].sum().item()
                if real_count == 0 or kept_count / real_count < args.min_kept_fraction:
                    dropped_low_kept += 1
                    continue
                # Count non-O confident tokens (labels 1..4)
                skill_count = ((labels[i] >= 1) & (labels[i] <= 4)).sum().item()
                if skill_count < args.min_skill_tokens:
                    dropped_no_skill += 1
                    continue
                out_records.append({
                    "input_ids": input_ids[i].cpu(),
                    "attention_mask": attention_mask[i].cpu(),
                    "labels": labels[i].cpu(),
                })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(out_records, args.output)
    print(f"Saved {len(out_records)} pseudo-labeled examples to {args.output}")
    print(f"Dropped: {dropped_low_kept} low-coverage, {dropped_no_skill} no-skill-tokens")


if __name__ == "__main__":
    main()
