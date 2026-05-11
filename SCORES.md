# SCORES — Skill-Extraction NER Results Log

Running log of evaluation results on the SkillSpan task. New experiments are appended to the top of the **Run log** so the most recent state is always visible first. Each entry follows the template below — leave `TBD` for any field not measured yet.

## Entry template

```
## YYYY-MM-DD — <short change name>
- **Change:** what was modified (1 line)
- **Config:** layers / d_model / d_ff / loss / KD on/off / other knobs that moved
- **Eval set:** SkillSpan val (or test)
- **Metrics:** macro F1 = X.XX · P = X.XX · R = X.XX
- **Per-class F1:** B-Tech / I-Tech / B-Knowledge / I-Knowledge / O
- **Inference:** CPU __ ms · CUDA __ ms (only if measured)
- **Notes:** what improved/regressed and suspected reason
```

---

## Current baseline (as of 2026-05-10)

| Model | Params | Macro F1 |
|---|---|---|
| Student — pre-distillation baseline | ~5.5M | ~0.50 |
| **Student — current (post-KD, focal + CRF + KL)** | **~5.5M** | **~0.59** |
| Teacher — DistilBERT-base | 66M | ~0.74 |

Target: close the student↔teacher gap. Industry rule of thumb with good KD is ~95% of teacher quality at 10–20% of params, i.e. ~0.70 F1 for this setup.

---

## Run log

## 2026-05-11 — Iterative self-training (student+teacher ensemble pseudo-labels)
- **Change:** Round-2 pseudo-labeling using a teacher (0.7) + student (0.3) softmax-probability ensemble. New script `pseudo_label_ensemble.py`. Pseudo set regenerated: **788 records / 17,064 skill tokens** (vs. round-1's 801 / 21,936 — fewer skill tokens because the ensemble disagreement filters out cases where student and teacher diverge).
- **Config:** Step 1 (teacher-PCA) baseline + `pseudo_data_path="data/pseudo_v2.pt"`. Early-stopped at epoch 44/50.
- **How to run:** `python pseudo_label_ensemble.py --input data/unlabeled_jds.txt --output data/pseudo_v2.pt && python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.6283** (−0.004 vs. step 1's 0.6322) · Test macro F1 = **0.63** (flat) · Test P = 0.64 · Test R = 0.63
- **Per-class F1 (test):** O = 0.94 · B-Tech = 0.50 (−0.01) · I-Tech = 0.55 (—) · B-Knowledge = **0.64** (+0.01) · I-Knowledge = **0.52** (+0.01)
- **Notes:** **Iterative round was a wash.** Trade-off was clean but didn't pay: fewer pseudo skill tokens (17k vs 22k, ~22% reduction) for higher consensus quality. The signal-loss outweighed the quality-gain. Knowledge classes moved up +0.01 each, Tech moved down −0.01, test macro identical to round 1. Likely the student is now close enough to the teacher on agreement zones that ensembling adds little — and the ensemble's down-weighting of single-confident-teacher predictions discards useful supervision.

## 2026-05-11 — PCA from fine-tuned teacher embedding (step 1 of two-step run)
- **Change:** Re-ran `init_embedding_pca.py` against the fine-tuned BERT-base teacher's embedding (`./checkpoints/teacher_bert_best_model.pt`) instead of the vanilla pretrained model. Wrote to `data/bert_embedding_pca128_teacher.pt`. `init_embedding_pca.py` extended with `--finetuned_ckpt` arg.
- **Config:** Previous run + teacher-PCA-init embedding (instead of vanilla-PCA-init).
- **How to run:** `python init_embedding_pca.py --finetuned_ckpt ./checkpoints/teacher_bert_best_model.pt --output_path data/bert_embedding_pca128_teacher.pt && python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.6322** (+0.005 vs. vanilla PCA 0.6276) · Test macro F1 = **0.63** (flat) · Test P = 0.63 · Test R = **0.64** (+0.02 vs. vanilla)
- **Per-class F1 (test):** O = 0.94 · B-Tech = **0.51** (+0.02) · I-Tech = 0.55 (—) · B-Knowledge = 0.63 (−0.01) · I-Knowledge = 0.51 (—)
- **Notes:** Mixed but slightly positive. Val and recall improved (B-Tech jumped +0.02). The task-adapted embedding directions help Tech specifically; Knowledge slightly regressed — possibly because the teacher's Knowledge fine-tuning was less consistent than its Tech adjustments. Explained variance after PCA was identical to vanilla (40.32%) because fine-tuning doesn't change BERT's eigenvalue spectrum much; the eigenvectors are what move.

## 2026-05-11 — PCA-initialized student embedding
- **Change:** Initialize the student's `embedding.weight` from PCA-projected `bert-base-uncased` word embeddings (30522×768 → 30522×128, rescaled to unit std). New script `init_embedding_pca.py`; `train.py` accepts `init_embedding_path` (default `data/bert_embedding_pca128.pt`) and copies the tensor into the model before training. Inference unaffected — same checkpoint format and parameter count.
- **Config:** All previous KD signals on (logit + hidden + attention KD, BERT-base teacher, 801 pseudo records) + PCA-init embedding. Early-stopped at epoch 26/50 (much faster than the 38–43 of recent runs — better starting point).
- **How to run:** `python init_embedding_pca.py && python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.6276** (+0.025) · Test macro F1 = **0.63** (+0.02) · Test P = 0.64 (+0.02) · Test R = 0.62 (+0.02) · Accuracy = 0.89 (+0.01)
- **Per-class F1 (test):** O = 0.94 · B-Tech = 0.49 (—) · I-Tech = **0.55** (+0.01) · B-Knowledge = **0.64** (+0.02) · **I-Knowledge = 0.51** (+0.05, biggest single-class jump in the entire optimization run)
- **Inference:** 11.28s total CPU
- **Notes:** **Biggest single test-set gain since logit KD itself.** The embedding is ~70% of the student's params (3.9M of 5.5M); starting it from PCA-projected BERT semantics retains 40.32% of BERT's word-vector variance while matching the student's 128-dim budget exactly. I-Knowledge moving +0.05 confirms the diagnostic earlier — that class was data/signal-limited, and a better word-level prior gives the encoder a head start it couldn't recover from random init. Faster convergence (26 vs 38 epochs) is a free bonus.

## 2026-05-11 — Attention-map KD added
- **Change:** Added TinyBERT-style attention-map distillation. `SelfAttention`, `TransformerBlock`, and `SkillExtractor` now optionally return per-layer attention probabilities (`return_attentions=True`). Teacher loaded with `attn_implementation="eager"` (default SDPA returns `None` for attentions). Loss: MSE between student attention maps and teacher attention maps with mean-over-heads alignment (student 4 heads, teacher 12 heads → mean → 1-vs-1) and the same layer subsampling as hidden KD. Computed in `autocast(enabled=False)` for numerical stability. `attention_kd_weight=10.0` (defaults to a large value because softmax-attn MSE is tiny in magnitude).
- **Config:** Previous run + attention-KD loss term. Early-stopped at epoch 39/50.
- **How to run:** `python train.py` (no flag changes; defaults enable attention KD)
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.6031** (−0.007 vs. previous 0.6101) · Test macro F1 = **0.61** (flat) · Test P = 0.63 · Test R = 0.60
- **Per-class F1 (test):** O = 0.94 · B-Tech = 0.49 · I-Tech = 0.54 · B-Knowledge = **0.62** (+0.01) · I-Knowledge = 0.46 (−0.01)
- **Notes:** **Attention KD did not help, slight regression on val.** Hypotheses for the muted/negative result: (a) `attention_kd_weight=10.0` may be too aggressive — likely needs a sweep over 0.1–10; (b) mean-over-heads alignment between 4-head student and 12-head teacher is lossy — head-pair-permutation matching may preserve more structure; (c) attention maps at shallow layers (0–2) may transfer near-embedding noise rather than the deep semantic patterns we want. Two debugging incidents along the way: (i) initial run crashed with `NoneType` because runtime `output_attentions=True` kwarg was ignored by some HF versions — fix: set `teacher_model.config.output_attentions=True` explicitly; (ii) crashed with `output_attentions not supported when attn_implementation=sdpa` — fix: load teacher with `attn_implementation="eager"`.

## 2026-05-10 — Bigger pseudo corpus relabeled by BERT teacher
- **Change:** Stacked two HF datasets (`jacob-hugging-face/job-descriptions` + `cnamuangtoun/resume-job-description-fit:job_description_text`) → 1082 unique JDs (vs. 806 before). Re-pseudo-labeled with the new BERT-base teacher. `prepare_unlabeled.py` extended with `--hf_datasets` for source stacking. `pseudo_label.py` updated to use BERT-base. New pseudo set: **801 records, 21,936 confident skill tokens** (vs. previous 586/12,896 — **+70% skill-token signal**).
- **Config:** Same as previous BERT-teacher run, only `data/pseudo.pt` regenerated. Merged training size 5601 (vs. 5386 previously).
- **How to run:** `python prepare_unlabeled.py --hf_datasets "jacob-hugging-face/job-descriptions,cnamuangtoun/resume-job-description-fit:job_description_text" --max_n 10000 && python pseudo_label.py --input data/unlabeled_jds.txt --output data/pseudo.pt && python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.6101** (+0.013) · Test macro F1 = **0.61** (flat) · Test P = 0.62 · Test R = 0.60 · Early-stopped at epoch 38/50.
- **Per-class F1 (test):** O = 0.94 · B-Tech = 0.49 · I-Tech = **0.54** (+0.02) · B-Knowledge = 0.61 (+0.01) · I-Knowledge = 0.47 (−0.01)
- **Inference:** 9.80s total CPU
- **Notes:** **Val gain is real (+0.013), test gain is essentially noise (rounded F1 same to 2 dp).** I-Tech moved most (+0.02), validating that pseudo-labels help long-span continuations specifically — same class the BERT-base teacher fixed by 0.02 over DistilBERT. **The original "10× more pseudo data" goal didn't pan out** — public HF datasets with raw JD free-text are scarce; we got 1.36× (1082 unique vs. 806). The bigger marginal effect is on val (cleaner labels, better teacher), not test. Total optimization stack now: 0.50 (baseline) → 0.59 (KD) → 0.60 (pseudo) → 0.61 (BERT teacher) → 0.61 (+ bigger pseudo). **Test F1 has been at 0.61 for two consecutive runs — likely close to the practical ceiling for this student architecture on this dataset without further architectural changes.**

## 2026-05-10 — Teacher upgrade: DistilBERT → BERT-base + hidden-KD fixes
- **Change:** Upgraded teacher from DistilBERT (66M, 6 layers) to BERT-base-uncased (110M, 12 layers). Teacher trained in two phases: v1 (5 ep, plain) hit 0.7436 val F1; **v2 (15 ep, weight_decay=0.1, label_smoothing=0.05, dropout=0.2, warmup+cosine, patience=4) hit 0.7612 val F1** — clear regularization win. Student-side changes: hidden-KD layer subsampling (student layer i ↔ teacher layer round(i·(T-1)/(S-1)) to cover the full 12-layer teacher depth; `hidden_kd_weight` 1.0 → 0.1; hidden-MSE computation moved to `torch.amp.autocast(enabled=False)` after fp16 overflow silently killed grad updates for two runs; added finite-loss assertion; clip_grad_norm now covers projector params too.
- **Config:** Same student arch (6L / d_model=128 / d_ff=768 / CRF) + new BERT-base teacher + fp32 hidden KD + pseudo data still merged.
- **How to run:** `python teacher_model.py && python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.5967** (vs. Fix #3 0.5920, **+0.005**) · Test macro F1 = **0.61** (+0.01) · Test P = 0.62 · Test R = 0.60 · Accuracy = 0.88. Early-stopped at epoch 43/50.
- **Per-class F1 (test):** O = 0.94 · B-Tech = **0.49** (+0.02) · I-Tech = 0.52 (—) · B-Knowledge = **0.60** (+0.01) · I-Knowledge = **0.48** (+0.01)
- **Teacher F1 (test, BERT v2):** macro 0.75 · B-Tech 0.69 · I-Tech 0.69 · B-Knowledge 0.77 · I-Knowledge 0.64
- **Inference:** 10.56s total test CPU (architecturally identical)
- **Notes:** **Smaller than expected (+0.01 vs. hoped +0.02–0.04).** Hypotheses for muted lift: (1) BERT teacher only +0.018 better than DistilBERT on val, so 50% capture rate (+0.009 → student) is reasonable; (2) pseudo dataset still labeled by *old* DistilBERT teacher — pseudo records carry stale signal; (3) `hidden_kd_weight` dropped 10× in this run, so the hidden-KD contribution actually dropped vs. the prior baseline. **Most promising cheap next move: re-generate pseudo labels with the new BERT-base teacher.** Two debugging incidents along the way (val loss frozen at 4 decimal places for 11 epochs each, twice) — root cause was BERT-base hidden-state magnitudes overflowing fp16 in the MSE term, producing inf gradients that GradScaler silently skipped. Now defended against with autocast-disabled fp32 path + assertion.

## 2026-05-10 — Fix #3: Pseudo-labeling on top of Fix #1
- **Change:** Augmented training data with 586 teacher-pseudo-labeled JDs (12.9k confident skill tokens) pulled from `jacob-hugging-face/job-descriptions` via `prepare_unlabeled.py`. Same model + KD config as Fix #1; `pseudo_data_path="data/pseudo.pt"` wired in via `ConcatDataset`.
- **Config:** Fix #1 + 586 extra pseudo records (merged train size 5386 vs. 4800 labeled-only). Pseudo gen: min_confidence=0.9, min_kept_fraction=0.5, min_skill_tokens=1. Early stopped at epoch 32/50 (vs. 45 before — converged faster).
- **How to run:** `python prepare_unlabeled.py && python pseudo_label.py --input data/unlabeled_jds.txt --output data/pseudo.pt && python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Best Val F1 = **0.5920** (vs. Fix #1 0.5810, **+0.011**) · Test macro F1 = **0.60** · Test P = 0.62 · Test R = 0.58 · Accuracy = 0.88
- **Per-class F1 (test):** O = 0.94 · B-Tech = **0.47** (+0.01) · I-Tech = **0.52** (+0.02) · B-Knowledge = **0.59** (+0.02) · I-Knowledge = **0.47** (+0.01)
- **Inference:** Total test inference 9.97s on CPU (slightly faster than Fix #1's 11.13s — likely warmup noise; arch is unchanged)
- **Notes:** **Modest but real lift across every skill class.** Recall moved more than precision (+0.01 on macro recall, +0.01 on macro P) — consistent with "more data → more skill tokens seen → finds more spans." Tech got the biggest absolute gain (I-Tech +0.02), supporting the diagnosis that Tech was data-limited. Smaller-than-hoped overall lift likely because pseudo set is only 586 examples (~12% data boost); a 5-10× larger unlabeled corpus would likely push further. Convergence at epoch 32 vs. 45 hints the extra signal was useful (model learned faster) rather than noisy.

## 2026-05-10 — Fix #2: Biaffine span head (no KD)
- **Change:** Replaced BIO+CRF with a biaffine span scorer over (start, end) token pairs → {None, Tech, Knowledge}. New module `BiaffineSpanHead`, model class `SpanSkillExtractor`, training script `train_biaffine.py`. Greedy non-overlapping decoding. KD intentionally not wired in v1.
- **Config:** 6 layers / d_model 128 / d_ff 768 / d_span_proj 64 / max_span_len 20 / weighted CE (None=0.1, Tech=1.0, Knowledge=1.0) / KD off
- **How to run:** `python train_biaffine.py`
- **Eval set:** SkillSpan val (best) + test
- **Metrics:** Best Val span F1 = **0.2816** · Test span P = 0.2530 · Test span R = 0.3591 · Test span F1 = **0.2969**
- **Per-class F1:** TBD (only macro reported; per-Tech/Knowledge breakdown not yet logged)
- **Inference:** TBD
- **Notes:** **NOT directly comparable to the 0.59 token-F1 above** — span F1 requires exact (start, end, class) match, much stricter. Recall > Precision (0.36 > 0.25): the model is over-predicting spans (mirror image of the BIO model's under-prediction). No KD, no teacher signal — vs. the BIO baseline which has KD, focal, CRF and class weights all working. **Need to compute span-F1 on the BIO model's predictions for a fair apples-to-apples baseline before drawing conclusions.** Likely tuning targets if we keep iterating: raise `none_weight` from 0.1 (over-prediction), add logit + hidden KD from teacher span pseudo-labels.

## 2026-05-10 — Fix #2 (retune): Biaffine with none_weight=0.3
- **Change:** Single-line tune: bumped `none_weight` from 0.1 → 0.3 in `train_biaffine.py`. Goal was to fix the over-prediction signal (R > P) seen in the previous biaffine run.
- **Config:** Same as previous biaffine run, only `none_weight=0.3`. Early stopped at epoch 46/50.
- **How to run:** `python train_biaffine.py`
- **Eval set:** SkillSpan val (best) + test (final)
- **Metrics:** Best Val span F1 = **0.3152** · Test span P = 0.3323 · Test span R = 0.3014 · **Test span F1 = 0.3161**
- **Per-class span F1 (test):** Tech = **0.1906** (P 0.2237 / R 0.1661) · Knowledge = **0.4169** (P 0.4044 / R 0.4302)
- **Notes:** Hypothesis on overall F1 confirmed: P went 0.25 → 0.33 (+0.08), R went 0.36 → 0.30 (−0.06), F1 went 0.30 → 0.32 (+0.02). **But the per-class story is the opposite of what I predicted.** I hypothesized biaffine would help Tech specifically (where BIO struggled at 0.24 due to B/I asymmetry). Reality: biaffine is **worse on Tech (0.19 vs 0.24)**, not better. The gap is also slightly larger on Tech (−0.052) than on Knowledge (−0.034). **Revised diagnosis:** the BIO B/I asymmetry was a symptom, not the cause. The real bottleneck on Tech is training signal (KD doing heavy lifting in BIO+CRF), not the labeling scheme. Biaffine without KD just lacks that signal uniformly — slightly more on Tech because Tech is the harder, more ambiguous class.

## 2026-05-10 — Span-F1 baseline of the BIO+CRF model (no retraining)
- **Change:** New script `evaluate_bio_as_spans.py` — loads the saved BIO+CRF student, CRF-decodes predictions on the test set, converts BIO sequences to spans, and scores with the same exact-match span-F1 metric the biaffine model uses. Apples-to-apples baseline.
- **Config:** Same Fix #1 checkpoint, no retraining. CRF Viterbi decoding.
- **How to run:** `python evaluate_bio_as_spans.py`
- **Eval set:** SkillSpan test (span-level)
- **Metrics:** Span P = 0.3766 · Span R = 0.3283 · **Span F1 = 0.3508**
- **Per-class span F1:** Tech = **0.2424** (P 0.2656 / R 0.2229) · Knowledge = **0.4505** (P 0.4749 / R 0.4284)
- **Notes:** Token-F1 0.59 → Span-F1 0.35 — confirms span-F1 is a much stricter metric (~24-point compression). Tech is the weak class (0.24); Knowledge is decent (0.45). Span Precision > Span Recall (under-predicts), matching the token-level pattern. **Direct comparison:** biaffine without any KD gets 0.30 — only 5 points behind a fully-tuned BIO+KD baseline. Biaffine has real headroom once KD is added.

## 2026-05-10 — Fix #1: Hidden-state distillation (hidden_kd_weight=1.0)
- **Change:** Added TinyBERT-style layer-wise hidden-state matching on top of existing logit KD. New `nn.Linear(128, 768)` projector projects student hiddens up to teacher size; masked MSE between projected student hidden states and DistilBERT's per-layer hidden states is added to the total loss. `SkillExtractor.forward` extended with `return_hidden_states` flag.
- **Config:** Same as current student + `hidden_kd_weight=1.0`, `teacher_hidden_size=768`. Layer matching: 1-to-1 over embedding output + each of the 6 transformer blocks (length 7 each side). Early stopped at epoch 45/50.
- **How to run:** `python train.py`
- **Eval set:** SkillSpan val (best) + test (per-class)
- **Metrics:** Val macro F1 = **0.5810** (best) · final-epoch Val F1 = 0.5724 · Val P = 0.6118 · Val R = 0.5411 · Test macro F1 = **0.59** · Test P = 0.61 · Test R = 0.57 · Accuracy = 0.88
- **Per-class F1 (test):** O = 0.94 · B-Tech = 0.46 · I-Tech = 0.50 · B-Knowledge = 0.57 · I-Knowledge = 0.46
- **Inference:** Total test inference 11.13s on CPU (per-batch latency TBD via benchmark_inference.py)
- **Notes:** **Effectively flat vs. baseline (~0.59).** Hidden KD did not help at this weight. Hypothesis: with `hidden_kd_weight=1.0` and a randomly-initialized 128→768 projector, the MSE term dominates early training and drowns the classification signal — the projector learns to satisfy MSE without forcing the encoder hiddens to be teacher-shaped. Per-class is most healthy on B-Knowledge (0.57); B-Tech and I-Knowledge are the weak spots. Worth retrying with `hidden_kd_weight=0.1` (or 0.3) before concluding hidden KD doesn't help here.

## 2026-05-10 — Current student (post-KD)
- **Change:** Knowledge distillation pipeline added: focal loss + CRF NLL + KL-divergence vs. teacher logits with temperature; deeper config.
- **Config:** 6 layers / d_model 128 / d_ff 768 / focal + CRF + KD / KD on (DistilBERT teacher)
- **Eval set:** SkillSpan val
- **Metrics:** macro F1 ≈ 0.59 · P = TBD · R = TBD
- **Per-class F1:** TBD
- **Inference:** CPU ~30 ms · CUDA TBD
- **Notes:** +9 F1 over pre-KD baseline. Residual issues: subword artifacts ("Dock", "Ku") in inference output, and category confusion ("Python" sometimes tagged Knowledge instead of Tech).

## 2026-05-10 — Pre-distillation baseline
- **Change:** Original main-branch student — plain CrossEntropyLoss, no CRF, no teacher.
- **Config:** 4 layers / d_model 128 / d_ff 512 / CE only / KD off
- **Eval set:** SkillSpan val
- **Metrics:** macro F1 ≈ 0.50 · P = TBD · R = TBD
- **Per-class F1:** TBD
- **Inference:** TBD
- **Notes:** Reference point before the fine-tune-approach branch's improvements.

## 2026-05-10 — Teacher (reference ceiling)
- **Change:** Standalone DistilBERT-base-uncased teacher trained for KD.
- **Config:** DistilBERT-base, 66M params / CE loss / linear warmup
- **Eval set:** SkillSpan val
- **Metrics:** macro F1 ≈ 0.74 · P = TBD · R = TBD
- **Per-class F1:** TBD
- **Inference:** TBD
- **Notes:** Frozen during student training; checkpoint at `./checkpoints/teacher_best_model.pt`. Sets the upper bound for the student under output-only KD.
