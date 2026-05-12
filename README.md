# Lightweight Skill Extractor

A 5.5M-parameter Transformer encoder, built from scratch in PyTorch, that extracts **Skill** and **Knowledge** spans from job descriptions. Trained via knowledge distillation from a 110M BERT-base teacher on the SkillSpan dataset. Runs on CPU in ~30 ms per job description.

## What this project does

Given a raw job description, the model returns two lists:

| Category | What it captures | Examples |
|---|---|---|
| **Skill** | Any ability or competence — technical or soft. | Python, Docker, Kubernetes, communication, leadership, mentoring, English |
| **Knowledge** | A body of theoretical or academic understanding. | linear algebra, statistics, consumer psychology, project management methodology |

The two labels follow the ESCO / SkillSpan taxonomy. **Skill** corresponds to ESCO's "Skill" class (renamed from "Tech" in earlier iterations, which had been misleading because the class covers soft skills too).

## What was achieved

Starting from a plain cross-entropy baseline at **0.50 macro F1**, four phases of work pushed the model to **0.63 macro F1** on the SkillSpan test set while keeping inference at ~30 ms on CPU:

| Phase | What was added | Test F1 |
|---|---|---|
| Baseline | Plain CE training | 0.50 |
| 1. Knowledge distillation | Logit + hidden + attention KD from BERT-base teacher | 0.59 |
| 2. Pseudo-labelling | Teacher labels ~800 unlabelled JDs | 0.60 |
| 3. Stronger teacher | Regularised BERT-base retrain | 0.61 |
| 4. PCA-init embedding | Initialise the student's 30,522 × 128 embedding from PCA-projected BERT word vectors | **0.63** |

A full write-up of each phase — including failed experiments — is in [REPORT.md](REPORT.md) and [SCORES.md](SCORES.md).

## Project structure

```
.
├── model.py                # SkillExtractor (encoder + CRF) + BiaffineSpanHead alternative
├── dataset.py              # SkillSpan loader + ID_TO_LABEL definition
├── train.py                # Full distillation training pipeline
├── teacher_model.py        # BERT-base teacher fine-tuning
├── pseudo_label.py         # Generate teacher pseudo-labels on unlabelled JDs
├── pseudo_label_ensemble.py# Round-2 pseudo-labelling via teacher+student ensemble
├── prepare_unlabeled.py    # Pull unlabelled JDs from HuggingFace datasets
├── init_embedding_pca.py   # PCA-project BERT's embedding to 128-dim
├── skill_taxonomy.py       # Hand-curated O*NET-style taxonomy used at inference
├── infer.py                # Production inference entry point
├── evaluate.py             # Token-level F1 evaluation
├── benchmark_inference.py  # CPU / CUDA latency benchmarking
├── diagnose_ceiling.py     # Diagnostic: measure teacher's own ceiling + label noise
├── REPORT.md               # Full project report
├── SCORES.md               # Chronological log of every training run
├── USAGE.md                # Inference usage guide
└── examples.md             # Nine worked-example JDs with model outputs
```

## Quickstart — running inference

### 1. Install dependencies

```bash
pip install torch transformers datasets scikit-learn tqdm numpy
```

### 2. Place a trained checkpoint at `./checkpoints/best_model.pt`

If you don't have one, train it (see *Training* below) or pull one from a release.

### 3. Run inference

```bash
python infer.py
```

The script runs against the sample JD baked into `infer.py`. To pass your own text, import the API directly:

```python
import torch
from infer import load_model, extract_skills

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model(device=device)

jd = """Senior DevOps Engineer with extensive experience in Kubernetes,
Docker, Terraform, and AWS. You will own CI/CD pipelines using Jenkins
and GitHub Actions. Strong scripting skills in Bash and Python required."""

result = extract_skills(jd, model, tokenizer, device=device)

print("Skills:")
for s in dict.fromkeys(result["Skill"]):
    print(f"  - {s}")

print("Knowledge:")
for s in dict.fromkeys(result["Knowledge"]):
    print(f"  - {s}")
```

### Example output

For the DevOps JD above:

```
Skills:
  - Kubernetes
  - Docker
  - Terraform
  - AWS
  - Jenkins
  - GitHub
  - Bash
  - Python

Knowledge:
  - CI
  - observability
```

More worked examples (ML, frontend, healthcare, marketing, etc.) are in [examples.md](examples.md).

## Training the model from scratch

End-to-end pipeline, in order:

```bash
# 1. Train the teacher (BERT-base, ~30-45 min on a single GPU)
python teacher_model.py

# 2. Optionally generate pseudo-labels from unlabelled JDs
python prepare_unlabeled.py --hf_datasets "jacob-hugging-face/job-descriptions"
python pseudo_label.py --input data/unlabeled_jds.txt --output data/pseudo.pt

# 3. Compute the PCA-projected embedding warm-start
python init_embedding_pca.py \
    --finetuned_ckpt ./checkpoints/teacher_bert_best_model.pt \
    --output_path data/bert_embedding_pca128_teacher.pt

# 4. Train the student with the full distillation stack
python train.py
```

## Architecture in one paragraph

The student is a 5.5M-parameter Transformer encoder: a 30,522 × 128 embedding table (BERT vocab), six pre-LayerNorm Transformer blocks (d_model=128, 4 heads, d_ff=768, GELU), a LayerNorm + Linear(128, 5) token classifier, and a linear-chain CRF with Viterbi decoding. The embedding table alone is 3.9M of the 5.5M parameters — a design asymmetry that made PCA-initialisation the most important late-stage intervention. The training objective sums focal loss, CRF NLL, KL-divergence logit KD, masked-MSE hidden-state KD with TinyBERT-style layer subsampling, and masked-MSE attention-map KD against a regularised BERT-base teacher.

## Further reading

- [USAGE.md](USAGE.md) — detailed inference guide with explanation of the taxonomy post-processor
- [REPORT.md](REPORT.md) — full project report (abstract, methodology, results, lessons)
- [SCORES.md](SCORES.md) — chronological experiment log with per-class F1 for every run
- [examples.md](examples.md) — nine worked-example JDs across multiple domains
