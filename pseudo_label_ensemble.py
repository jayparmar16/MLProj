"""Round-2 pseudo-labeling using an ensemble of the trained STUDENT and TEACHER.

Motivation: after the first round of distillation, the student has learned
useful task-specific patterns. Averaging its predictions with the teacher's
yields pseudo-labels that are higher-quality where the two agree and naturally
filtered where they disagree.

Ensemble: weighted average of softmax probs per token.
  ensemble_prob = teacher_weight * teacher_prob + student_weight * student_prob
  pseudo_label  = argmax(ensemble_prob)
  confidence    = max(ensemble_prob)
Only tokens whose confidence exceeds `--min_confidence` keep their predicted
label; the rest are masked (-100) so the student isn't forced to learn them.

Usage:
  python pseudo_label_ensemble.py \\
      --input data/unlabeled_jds.txt \\
      --output data/pseudo_v2.pt \\
      --student_ckpt ./checkpoints/best_model.pt \\
      --teacher_ckpt ./checkpoints/teacher_bert_best_model.pt
"""
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, BertForTokenClassification

from model import SkillExtractor


def load_teacher(ckpt_path, num_labels=5, device="cpu"):
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    ).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at {ckpt_path}.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_student(ckpt_path, vocab_size, device="cpu",
                 num_layers=6, d_ff=768, dropout=0.2, use_crf=True, max_len=128):
    model = SkillExtractor(
        vocab_size=vocab_size, d_model=128, num_heads=4,
        num_layers=num_layers, d_ff=d_ff, num_classes=5,
        max_len=max_len, dropout=dropout, use_crf=use_crf,
    ).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Student checkpoint not found at {ckpt_path}.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/pseudo_v2.pt")
    parser.add_argument("--teacher_ckpt", default="./checkpoints/teacher_bert_best_model.pt")
    parser.add_argument("--student_ckpt", default="./checkpoints/best_model.pt")
    parser.add_argument("--teacher_weight", type=float, default=0.7,
                        help="Teacher remains stronger (val F1 0.74 vs student 0.63), so it gets the larger share.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_confidence", type=float, default=0.85,
                        help="Slightly looser than round 1 because ensemble probs are smoother (averaging "
                             "soft distributions reduces peak confidence vs a single sharp teacher).")
    parser.add_argument("--min_kept_fraction", type=float, default=0.5)
    parser.add_argument("--min_skill_tokens", type=int, default=1)
    args = parser.parse_args()

    student_weight = 1.0 - args.teacher_weight
    print(f"Ensemble weights — teacher: {args.teacher_weight}, student: {student_weight:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher = load_teacher(args.teacher_ckpt, device=device)
    student = load_student(args.student_ckpt, vocab_size=tokenizer.vocab_size, device=device)
    print(f"Loaded teacher from {args.teacher_ckpt}")
    print(f"Loaded student from {args.student_ckpt}")

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"Read {len(lines)} unlabeled descriptions from {args.input}.")

    out_records = []
    dropped_low_kept = 0
    dropped_no_skill = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(lines), args.batch_size), desc="Ensemble pseudo-labeling"):
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

            t_logits = teacher(input_ids, attention_mask=attention_mask).logits
            s_logits = student(input_ids, attention_mask)

            t_probs = t_logits.softmax(dim=-1)
            s_probs = s_logits.softmax(dim=-1)
            ens_probs = args.teacher_weight * t_probs + student_weight * s_probs
            conf, preds = ens_probs.max(dim=-1)

            real = attention_mask.bool()
            confident = conf >= args.min_confidence
            keep = real & confident
            labels = preds.masked_fill(~keep, -100)

            for i in range(input_ids.size(0)):
                real_count = real[i].sum().item()
                kept_count = keep[i].sum().item()
                if real_count == 0 or kept_count / real_count < args.min_kept_fraction:
                    dropped_low_kept += 1
                    continue
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
    print(f"Saved {len(out_records)} ensemble-pseudo-labeled examples to {args.output}")
    print(f"Dropped: {dropped_low_kept} low-coverage, {dropped_no_skill} no-skill-tokens")


if __name__ == "__main__":
    main()
