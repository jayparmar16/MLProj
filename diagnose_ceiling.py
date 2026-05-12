"""Diagnose how much of the remaining gap is annotation noise vs. learnable headroom.

Three measurements, all run on the SkillSpan validation set:

1. Teacher token-level macro F1 — this is the ceiling for pure KD.
   The student cannot out-predict the teacher's distribution without extra data.
2. Per-class teacher / gold disagreement rate.
3. A sample of high-confidence teacher disagreements (teacher > 0.9 confident,
   gold says something else) — these are the cases where the teacher is willing
   to bet against the annotation. Eyeball them: if they look like gold-label
   errors or genuine ambiguity, the dataset ceiling is real.

Run:
  python diagnose_ceiling.py
"""
import os
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, BertForTokenClassification
from tqdm import tqdm

from dataset import get_dataloaders, ID_TO_LABEL


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(
    teacher_ckpt="./checkpoints/teacher_bert_best_model.pt",
    high_conf_threshold=0.9,
    sample_size=40,
):
    device = get_device()
    print(f"Device: {device}\n")

    # Student and teacher both use the bert-base-uncased vocab — same input_ids.
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=32)
    teacher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    teacher = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    ).to(device)
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint missing at {teacher_ckpt}.")
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    print(f"Loaded teacher from {teacher_ckpt}")

    # Collect predictions
    all_preds, all_golds, all_confs, all_tokens = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Teacher inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = teacher(input_ids, attention_mask=attention_mask).logits
            probs = logits.softmax(dim=-1)
            conf, preds = probs.max(dim=-1)

            for i in range(input_ids.size(0)):
                for j in range(input_ids.size(1)):
                    g = int(labels[i, j])
                    if g == -100:
                        continue
                    all_preds.append(int(preds[i, j]))
                    all_golds.append(g)
                    all_confs.append(float(conf[i, j]))
                    all_tokens.append(int(input_ids[i, j]))

    all_preds = np.array(all_preds)
    all_golds = np.array(all_golds)
    all_confs = np.array(all_confs)

    # ---- 1. Teacher macro F1 ----
    print("\n" + "=" * 60)
    print("  1. TEACHER F1 ON VAL — hard ceiling for pure KD")
    print("=" * 60)
    p, r, f, _ = precision_recall_fscore_support(
        all_golds, all_preds, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0
    )
    print(f"Teacher macro F1 on val: {f:.4f} (P {p:.4f} / R {r:.4f})")
    print("Student current macro F1: ~0.59")
    print(f"Headroom (pure KD): {f - 0.59:+.3f}")

    # Per-class teacher F1
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        all_golds, all_preds, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    print("\nPer-class teacher F1:")
    for i, lbl in ID_TO_LABEL.items():
        print(f"  {lbl:>14}: F1={f_c[i]:.4f}  P={p_c[i]:.4f}  R={r_c[i]:.4f}")

    # ---- 2. Disagreement rates by gold class ----
    print("\n" + "=" * 60)
    print("  2. TEACHER vs GOLD DISAGREEMENT RATE (by gold class)")
    print("=" * 60)
    for cls_id, cls_name in ID_TO_LABEL.items():
        mask = all_golds == cls_id
        if mask.sum() == 0:
            continue
        disagree = (all_preds[mask] != cls_id).mean()
        high_conf_disagree = ((all_preds[mask] != cls_id) & (all_confs[mask] >= high_conf_threshold)).sum()
        n = int(mask.sum())
        print(f"  {cls_name:>14}: {disagree*100:5.1f}% disagree "
              f"({high_conf_disagree:4d} of {n:5d} tokens — teacher >{high_conf_threshold:.2f} confident)")

    print(f"\nA high 'high-conf disagree' count means the teacher is willing to override "
          f"the annotation. Often a sign of label noise rather than student headroom.")

    # ---- 3. Sample of high-confidence disagreements ----
    print("\n" + "=" * 60)
    print(f"  3. SAMPLE: {sample_size} high-confidence ({high_conf_threshold:.2f}+) disagreements")
    print("=" * 60)
    print("  (token | gold | teacher predicts | confidence)")
    print("-" * 60)

    hc_mask = (all_preds != all_golds) & (all_confs >= high_conf_threshold)
    hc_idx = np.where(hc_mask)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(hc_idx, size=min(sample_size, len(hc_idx)), replace=False)

    for idx in pick:
        tok = teacher_tokenizer.convert_ids_to_tokens([all_tokens[idx]])[0]
        gold = ID_TO_LABEL[all_golds[idx]]
        pred = ID_TO_LABEL[all_preds[idx]]
        print(f"  {tok:>20s}  |  gold={gold:<12s}  |  teacher={pred:<12s}  |  {all_confs[idx]:.3f}")

    print("\n" + "=" * 60)
    print("  INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
If most high-confidence disagreements above look like:
  - Tokens that COULD reasonably be either class (Python = Tech or Knowledge?)
  - Tokens that are clearly mis-labeled in the gold
  → The dataset ceiling is real. Pushing past 0.70 requires data cleaning,
    not better models. Pseudo-labeling helps because it averages over noise.

If most look like:
  - Tokens where teacher is clearly wrong and gold is right
  → The teacher has more room to improve too. Re-training a bigger / better
    teacher first would lift the whole stack.

Either way: teacher F1 above is the practical ceiling for the student.
The student is currently at ~0.59 macro F1.
""")


if __name__ == "__main__":
    main()
