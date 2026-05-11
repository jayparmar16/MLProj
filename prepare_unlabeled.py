"""Pull a public job-description dataset from the HuggingFace Hub and write it
out as one description per line at data/unlabeled_jds.txt — the input format
that pseudo_label.py expects.

Default uses `jacob-hugging-face/job-descriptions`. If that dataset is gated,
removed, or has a different schema than expected, override via:
  --hf_dataset <name> --text_field <col> --split <split>

Tries to be schema-tolerant: if --text_field isn't given, picks the longest
average-length string column automatically.

Run:
  python prepare_unlabeled.py
  python prepare_unlabeled.py --hf_dataset some/other-jd-dataset --max_n 5000
"""
import argparse
import os
import re
import sys


def detect_text_field(sample_rows):
    """Pick the string column with the longest average length."""
    if not sample_rows:
        raise ValueError("No rows to inspect.")
    candidates = {}
    for row in sample_rows:
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 20:
                candidates.setdefault(k, []).append(len(v))
    if not candidates:
        raise ValueError(f"No long-text string columns found. Columns: {list(sample_rows[0].keys())}")
    best = max(candidates.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    return best[0]


def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="jacob-hugging-face/job-descriptions",
                        help="Single HF dataset (legacy). Ignored if --hf_datasets is used.")
    parser.add_argument("--hf_datasets", default=None,
                        help="Comma-separated list of HF datasets to stack. Optionally per-dataset "
                             "text_field via 'name:field' syntax. Example: "
                             "'jacob-hugging-face/job-descriptions,cnamuangtoun/resume-job-description-fit:job_description_text'")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default=None,
                        help="Column with the JD text. Auto-detected if omitted. "
                             "Ignored when --hf_datasets uses 'name:field' syntax.")
    parser.add_argument("--output", default="data/unlabeled_jds.txt")
    parser.add_argument("--max_n", type=int, default=10000,
                        help="Cap the number of JDs written. None to keep all.")
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=8000)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` not installed. Run: pip install datasets")

    # Resolve dataset list — either a single source (legacy) or stacked sources.
    sources = []
    if args.hf_datasets:
        for entry in args.hf_datasets.split(","):
            entry = entry.strip()
            if ":" in entry:
                name, field = entry.split(":", 1)
                sources.append((name.strip(), field.strip()))
            else:
                sources.append((entry, args.text_field))
    else:
        sources.append((args.hf_dataset, args.text_field))

    seen = set()
    written = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for name, requested_field in sources:
            if args.max_n and written >= args.max_n:
                break
            print(f"\nLoading {name} (split={args.split})…")
            ds = load_dataset(name, split=args.split)
            print(f"  {len(ds)} rows. Columns: {list(ds.features)}")
            text_field = requested_field or detect_text_field([ds[i] for i in range(min(50, len(ds)))])
            print(f"  Using text field: '{text_field}'")

            kept_from_source = 0
            for row in ds:
                text = row.get(text_field) or ""
                text = clean(text)
                if not (args.min_chars <= len(text) <= args.max_chars):
                    continue
                key = text[:200]
                if key in seen:
                    continue
                seen.add(key)
                f.write(text + "\n")
                written += 1
                kept_from_source += 1
                if args.max_n and written >= args.max_n:
                    break
            print(f"  Kept {kept_from_source} unique JDs from this source. Running total: {written}.")

    print(f"\nWrote {written} unlabeled JDs to {args.output}")
    print("\nNext:")
    print(f"  python pseudo_label.py --input {args.output} --output data/pseudo.pt")


if __name__ == "__main__":
    main()
