# Skill-Extraction NER on a 5.5M-Parameter Budget — Project Report

## Abstract

We built and optimized an ultra-lightweight Transformer encoder (~5.5M parameters) for Named Entity Recognition on the SkillSpan dataset, extracting Tech and Knowledge skill spans from job descriptions via BIO tagging. Starting from a baseline of 0.50 macro F1 on test, we systematically applied knowledge distillation, pseudo-labeling, architectural experiments, and pretrained-embedding warm-starts to lift performance to **0.63 macro F1** — a 26% relative improvement, while keeping CPU inference under ~30 ms per job description. A stretch target of 0.70 F1 proved infeasible at the 5.5M-parameter budget on this dataset; the practical ceiling is bound by both the teacher's own performance (~0.74) and SkillSpan's annotation noise. The single biggest underestimated lever turned out to be initializing the student's word-embedding table with PCA-projected BERT vectors (+0.02 test F1, after sophisticated KD methods had plateaued).

## Deliverables

### Code

A complete training, inference, and diagnostic pipeline for compact Transformer NER, built on top of the original `feat/lightweight-skill-extractor` branch. New files and substantial edits:

- **[model.py](model.py)** — added `CRF` layer (linear-chain CRF with Viterbi decoding), `BiaffineSpanHead`, `SpanSkillExtractor`, plus `return_hidden_states` and `return_attentions` flags on the encoder for layerwise distillation.
- **[train.py](train.py)** — training pipeline: focal loss, CRF NLL, logit KD, hidden-state KD (TinyBERT-style with layer subsampling), attention-map KD, pseudo-data merging, PCA-init embedding loading, fp32-stable computation paths, non-finite-loss assertions.
- **[teacher_model.py](teacher_model.py)** — upgraded from DistilBERT (66M) to regularized BERT-base (110M) teacher: weight decay 0.1, label smoothing 0.05, dropout 0.2, warmup+cosine LR, early stopping.
- **[pseudo_label.py](pseudo_label.py)** — generates teacher-pseudo-labeled records from unlabeled JDs with confidence filtering.
- **[pseudo_label_ensemble.py](pseudo_label_ensemble.py)** — round-2 pseudo-labeling using a teacher+student softmax-probability ensemble.
- **[prepare_unlabeled.py](prepare_unlabeled.py)** — pulls and dedupes JD text from one or more HuggingFace datasets.
- **[init_embedding_pca.py](init_embedding_pca.py)** — projects BERT's 30522×768 embedding to 30522×128 via PCA, with optional fine-tuned-teacher source.
- **[diagnose_ceiling.py](diagnose_ceiling.py)** — diagnostic measuring teacher F1, per-class disagreement rates, and a sample of high-confidence teacher↔gold mismatches.
- **[evaluate_bio_as_spans.py](evaluate_bio_as_spans.py)** and **[evaluate_biaffine_per_class.py](evaluate_biaffine_per_class.py)** — span-level F1 evaluators that made apples-to-apples comparison between BIO+CRF and biaffine heads possible.
- **[train_biaffine.py](train_biaffine.py)** — alternative training script for the biaffine span head.
- **[benchmark_inference.py](benchmark_inference.py)** — CPU/CUDA latency benchmarking.
- **[infer.py](infer.py)** — production inference: takes raw text, returns `{"Tech": [...], "Knowledge": [...]}` spans via tokenizer offset mapping.

### Trained model weights

- **`./checkpoints/best_model.pt`** — final student, ~5.49M params, **0.63 macro F1 on test** (up from 0.50 baseline).
- **`./checkpoints/teacher_bert_best_model.pt`** — regularized BERT-base teacher, 110M params, 0.7612 best val F1 / 0.75 test macro F1.

### Processed data artifacts

- **`data/pseudo.pt`** — 801 teacher-pseudo-labeled JD examples (21,936 confident skill tokens).
- **`data/pseudo_v2.pt`** — 788 ensemble-pseudo-labeled examples (17,064 skill tokens).
- **`data/unlabeled_jds.txt`** — 1,082 unique JDs stacked and deduped from two public HF datasets.
- **`data/bert_embedding_pca128.pt`** and **`data/bert_embedding_pca128_teacher.pt`** — PCA-projected embedding tensors used to warm-start the student.

### Documentation

- **[SCORES.md](SCORES.md)** — chronological log of every experiment with config, metrics, per-class breakdown, and post-hoc interpretation including failures.
- **[USAGE.md](USAGE.md)** — inference usage notes for downstream consumers.

## Experiment design

When the project started, the goal was vague: "improve the lightweight skill extractor." The first deliberate decision was to set up [SCORES.md](SCORES.md) as a running log — every experiment would record (1) what changed, (2) test macro F1, (3) per-class F1, and (4) honest interpretation including failures. This turned out to be the single most important decision in the project. It forced every subsequent step to have a measurable hypothesis and outcome, and revealed patterns (val-vs-test divergence, class-specific ceilings) that would have been invisible in a series of unaudited training runs.

The primary success metric was **macro F1** on the SkillSpan test set across 5 classes (O, B-Tech, I-Tech, B-Knowledge, I-Knowledge). We chose macro over weighted F1 because skill tokens are heavily outnumbered by O tokens (~85%), and weighted F1 lets the model trivially succeed by predicting "no skill" everywhere — uninformative for our task.

The stretch goal of **0.70 macro F1** was set partway through the project, after the first run of `diagnose_ceiling.py` measured the teacher's own ceiling at 0.7294 on val. Hitting 0.70 with a 5.5M-param student would mean reaching ~95% of teacher quality at 12× fewer parameters — the TinyBERT/MiniLM ballpark, but at less than half their parameter count.

Three design decisions shifted during the project that are worth flagging:

1. **A span-level F1 metric was added mid-project.** When we trialled a biaffine span head, we couldn't compare directly to the token-level F1 of BIO+CRF — span F1 is much stricter (exact-match start, end, class). We wrote `evaluate_bio_as_spans.py` to run BIO predictions through the span F1 metric. Without this, we would have falsely concluded the biaffine approach was much worse than it actually was (the metric difference alone accounted for most of the 0.30 vs 0.59 gap).

2. **Diagnostic measurements were instrumented before later interventions.** `diagnose_ceiling.py` measured the teacher's per-class F1 and high-confidence disagreement rates with gold labels. This told us upper bounds for each class — and that ~30% of high-confidence teacher predictions on Tech I-tags disagreed with gold — before we spent more compute on architectural experiments that wouldn't have helped.

3. **The 5.5M-param budget was treated as a hard constraint.** We considered relaxing it (to a TinyBERT-class 14M or wider d_model=192/256 student) and explicitly chose not to. This rejected an otherwise-attractive +0.03–0.07 expected gain and kept the optimization in compression-research territory.

## Methodology

### Base architecture

A Transformer encoder built from scratch in PyTorch — not a fine-tune of an existing checkpoint:
- Embedding: 30522 × 128 (BERT vocab, learned weights — 70% of total parameters)
- 6 transformer blocks: d_model=128, 4 heads, d_ff=768, pre-LN, GELU
- LayerNorm + Linear(128, 5) classifier
- Linear-chain CRF head (5×5 transition matrix + start/end vectors)

Total trainable parameters: ~5.49M.

The encoder is intentionally narrow (d_model=128 is unusual — most public small Transformers use 192–384). Param budget forced this. Multi-head attention uses 4 heads at d_k=32 each. Sinusoidal position encoding (non-trainable) rather than learned positions.

### Loss composition

The final training objective sums five terms with the listed weights:
- **Focal loss** on token logits (weight 0.5) — handles ~85% O-class imbalance via class weights derived from inverse frequency.
- **CRF negative log-likelihood** (weight 1.0) — enforces valid BIO transitions at the sequence level.
- **KL divergence** between student and teacher logits at temperature 2.0 (weight 0.5) — soft knowledge distillation.
- **Masked MSE** between student per-layer hidden states (projected via a learnable Linear(128, 768)) and teacher per-layer hidden states (weight 0.1) — TinyBERT-style feature matching with layer subsampling (student's 6 layers matched to teacher's 12 via uniform sampling).
- **Masked MSE** between student averaged-over-heads attention maps and teacher averaged-over-heads attention maps (weight 10.0 — large because attention-map MSE values are tiny in absolute terms) — attention transfer.

The weight balance is delicate. Hidden KD originally started at weight 1.0 and worked fine with the DistilBERT teacher. When we upgraded to BERT-base, the hidden-state magnitudes were larger; the MSE overflowed fp16 inside autocast; gradients went infinite; `GradScaler` silently skipped every step for 11 epochs. Fix: drop the weight to 0.1 and force the MSE computation into fp32 by wrapping it in `torch.amp.autocast(enabled=False)`. We also added `assert torch.isfinite(loss)` to surface this kind of silent failure earlier.

### Teacher

A `BertForTokenClassification` fine-tuned on SkillSpan. After a plain 5-epoch fine-tune capped at 0.7436 val F1 with clear overfitting (train loss collapsing while val plateaued), a regularized retrain pushed the teacher to **0.7612 best val F1, 0.75 test macro F1**:
- 15 epochs with patience=4 early stopping (triggered at epoch 11)
- weight decay 0.1 (up from 0.01)
- label smoothing 0.05
- hidden, attention, and classifier dropout all 0.2
- warmup+cosine LR (peak 3e-5)

### Data

**SkillSpan** (`jjzha/skillspan` on HuggingFace) provides labeled splits — ~4,800 training, 588 val, ~1,200 test examples. We augmented training with two pseudo-labeling rounds:

- **Round 1**: 801 examples from 1,082 unique JDs pulled from `jacob-hugging-face/job-descriptions` (806 unique) and `cnamuangtoun/resume-job-description-fit` (276 unique). The trained BERT teacher labeled each token; tokens with softmax max prob < 0.9 were masked (-100); sequences with too few confident skill tokens were dropped.
- **Round 2**: 788 examples re-labeled by a 0.7-teacher / 0.3-student softmax-probability ensemble. Confidence threshold lowered to 0.85 because ensembling smooths peak confidence.

We had hoped to pull 10× more unlabeled JDs but found that public HF datasets with raw free-text JDs are scarce; the larger `lukebarousse/data_jobs` (785k rows) only contains structured skill extracts, not full text. We ended up with 1.36× the original 806, not 10×.

### Embedding warm-start

Project the trained BERT-base teacher's 30522 × 768 word embedding to 30522 × 128 via PCA (retains 40.32% of variance), rescale to unit per-element std to match `nn.Embedding`'s default init scale, and copy into the student's embedding table before training. Since the embedding is 3.9M of the student's 5.5M parameters, a non-random starting point gives the encoder dramatically more useful inputs from epoch 0.

### Experiments not adopted

- **Biaffine span head**: A `SpanSkillExtractor` class scoring every (start, end) token pair as one of {None, Tech, Knowledge} was implemented and trained via `train_biaffine.py`. With `none_weight=0.3` and no KD, span F1 reached 0.32 — vs. BIO+CRF's 0.35 on the same span-F1 metric. Competitive without any KD signal, but didn't decisively beat the tagging approach. We kept the code but did not use it for the production model.
- **Wider student architectures** (d_model=192/256, 14M-class TinyBERT replica): considered and explicitly rejected to preserve the 5.5M-param budget.
- **Attention KD weight sweep**: After attention KD at weight 10.0 added 0.00 to test F1, we did not sweep the weight down. The expected return was below 0.01.

## Results

### Final numbers

On the SkillSpan test set with the production checkpoint:

| Metric | Value |
|---|---|
| **Macro F1** | **0.63** |
| Macro precision | 0.64 |
| Macro recall | 0.62 |
| Accuracy | 0.89 |
| Best val F1 | 0.6322 |
| CPU inference latency | ~30 ms / 100-word JD |

Per-class F1: O = 0.94 · B-Tech = 0.51 · I-Tech = 0.55 · B-Knowledge = 0.64 · I-Knowledge = 0.51.

### Progression

| Stage | Test F1 | Δ |
|---|---|---|
| Baseline (plain CE) | 0.50 | — |
| + Logit KD (DistilBERT) | 0.59 | **+0.09** |
| + Hidden KD | 0.59 | 0.00 |
| + Pseudo-labels (586) | 0.60 | +0.01 |
| + BERT-base teacher | 0.61 | +0.01 |
| + Bigger pseudo (801) | 0.61 | 0.00 |
| + Attention KD | 0.61 | 0.00 |
| **+ PCA embedding init** | **0.63** | **+0.02** |
| + Teacher-PCA + iterative self-training | 0.63 | 0.00 |
| Stretch target | 0.70 | — |

Macro recall improved more than macro precision over the trajectory (≈0.50 → 0.62 vs ≈0.50 → 0.64). The model became substantially more willing to predict skill spans without losing precision — a healthier direction than the inverse.

### Uncertainties when interpreting these results

- **Single-seed runs.** Each row above is one training run, not a multi-seed mean. Observed run-to-run variance on this task is roughly ±0.005–0.01 macro F1. The 0.61 → 0.61 plateaus may include underlying small movement that's hidden by rounding to two decimal places.
- **Test/val divergence.** Some changes (attention KD especially) improved val without moving test, hinting at partial overfitting to validation-distribution patterns. The test set was held out throughout — never used for early stopping or hyperparameter selection — but we observed that val F1 sometimes moved 1–2 points further than test on the same change.
- **The 0.74 teacher is an upper bound.** Pure knowledge distillation cannot exceed the teacher's distribution. Pseudo-labels are also teacher-derived, so they don't add independent supervision against this bound — they just spread the teacher's signal across more examples.
- **SkillSpan annotation noise.** The `diagnose_ceiling.py` run showed ~30% of skill tokens have high-confidence (>0.9) teacher disagreement with gold. Some of those are teacher errors; a meaningful fraction look like genuine label ambiguity (e.g., "communication" tagged B-Tech vs B-Knowledge in different contexts). The dataset itself caps the achievable F1, independent of the model.

## Lessons learned

**1. Set up the measurement scaffold before optimizing anything.** The single most valuable artifact was [SCORES.md](SCORES.md). Forcing every experiment to log config, metrics, and interpretation surfaced patterns that would have been invisible across many one-off runs. The cost was negligible (~5 minutes per experiment). Without it, several "this seems better" decisions would have gone unaudited and conclusions about which interventions actually helped would have been guesswork.

**2. Diagnose before architecting.** We almost committed to a biaffine span head based on a partial diagnosis ("BIO+CRF has B/I asymmetry on Tech, so the labeling scheme is the problem"). Writing the 100-line `diagnose_ceiling.py` revealed the dominant issue was the teacher's own weakness on multi-word skill continuations — not the labeling scheme. The script took 30 minutes to write and saved days of misdirected effort. Cheap diagnostic scripts pay for themselves the first time they redirect a major decision.

**3. Silent failures are the worst kind.** Twice during the BERT-teacher upgrade, training "completed" with val F1 frozen to four decimal places across 11 epochs. The cause both times was fp16 overflow in the hidden-KD MSE → infinite gradients → `GradScaler` silently skipping every step. Without the silent-failure pattern (suspiciously constant val loss), we would have assumed the model just hadn't learned. After the second incident we added an explicit non-finite-loss assertion in the training loop. **Anything that can break silently should be made to break loudly.**

**4. Cheap initialization tricks can beat sophisticated KD recipes.** Attention-map KD is well-studied (TinyBERT, MiniLM), required ~150 lines of code, and added 0.00 to test F1. PCA-projecting BERT's embedding into the student's 128-dim space is a 30-line one-shot script — and added +0.02, the biggest gain in the post-KD half of the project. The lesson isn't "skip distillation"; it's that for compact models where embedding params dominate, the embedding initialization is much higher-leverage than people assume.

**5. Negative results matter and should be logged.** Roughly half the experiments produced no measurable improvement. Each was informative: which lever was already saturated (hidden KD beyond a point), which architectural assumption was wrong (biaffine isn't always better at boundary errors), which class was data-bound rather than model-bound (Tech I-tags). Without logging the failures alongside the successes, we would have looped back to them.

**6. Honest ceilings beat false optimism.** The 0.70 target was aspirational. After multiple experiments converged at 0.63, the right call was to stop, document, and ship — not to chase further +0.005 improvements through hyperparameter sweeps. Recognizing diminishing returns is itself a project skill. The trajectory log made this judgment call evidence-based rather than vibes-based.

**7. Hard constraints force genuinely different solutions.** Most of what worked here — PCA-init embedding, regularized teacher upgrade, layer-subsampled hidden KD — wouldn't have been needed at 50M+ parameters. The 5.5M-param ceiling forced novel choices that wouldn't have surfaced otherwise. Hard constraints are worth setting deliberately, not just accepting.

**8. The dataset is usually the bottleneck, eventually.** The diagnostic ceiling analysis identified ~30% high-confidence label disagreement on Tech I-tags. Past a certain point, model improvements just retrain over the same noise. The next investment that would actually move test F1 is not better training tricks — it's cleaner or more labels. That's a useful conclusion to reach explicitly, even if we didn't act on it within this project's scope.
