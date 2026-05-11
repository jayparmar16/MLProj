import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math
import numpy as np
from collections import Counter

from dataset import get_dataloaders, ID_TO_LABEL
from model import SkillExtractor
from evaluate import evaluate_model, print_detailed_metrics
from transformers import BertForTokenClassification


# ---------------------------------------------------------------------------
# Focal Loss  –  handles extreme class imbalance (85 % of tokens are "O")
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class Focal Loss with per-class weighting and label smoothing."""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        num_classes = logits.size(-1)

        # Label smoothing: soften the one-hot targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets_one_hot = targets.unsqueeze(1)
        log_p_t = log_probs.gather(1, targets_one_hot).squeeze(1)
        p_t = probs.gather(1, targets_one_hot).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        if self.label_smoothing > 0:
            loss = -(focal_weight.unsqueeze(1) * smooth_targets * log_probs).sum(dim=-1)
        else:
            loss = -focal_weight * log_p_t

        return loss.mean()


def compute_class_weights(train_loader, num_classes=5):
    """Compute inverse-frequency class weights from training data."""
    print("Computing class weights from training data...")
    label_counts = Counter()
    for batch in tqdm(train_loader, desc="Scanning labels", leave=False):
        labels = batch["labels"].view(-1)
        for label in labels.tolist():
            if label != -100:
                label_counts[label] += 1

    total = sum(label_counts.values())
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        w = total / (num_classes * count)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()

    print("Class weights:")
    for i, w in enumerate(weights):
        print(f"  {ID_TO_LABEL[i]:>13}: {w:.4f}  (count: {label_counts.get(i, 0):,})")

    return weights


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train(
    epochs=50,
    batch_size=32,
    lr=5e-4,
    max_length=128,
    save_dir="./checkpoints",
    warmup_ratio=0.1,
    accum_steps=2,
    patience=10,
    focal_gamma=2.0,
    label_smoothing=0.05,
    # --- Model architecture ---
    num_layers=6,
    d_ff=768,
    dropout=0.2,
    use_crf=True,
    crf_loss_weight=1.0,
    focal_loss_weight=0.5,
    # --- KD parameters ---
    teacher_model_path="./checkpoints/teacher_bert_best_model.pt",
    temperature=2.0,
    kd_weight=0.5,
    # --- Hidden-state KD ---
    hidden_kd_weight=0.1,
    teacher_hidden_size=768,
    # --- Attention-map KD (TinyBERT-style) ---
    attention_kd_weight=10.0,  # MSE on 128x128 attention probs is small numerically; needs scale.
    # --- Pseudo-labeled data ---
    pseudo_data_path=None,
    # --- PCA-projected BERT embedding init (cheap warm-start) ---
    init_embedding_path=None,
):
    device = get_device()
    print(f"{'='*60}")
    print(f"  SKILL EXTRACTOR — KNOWLEDGE DISTILLATION")
    print(f"{'='*60}")
    print(f"Device:              {device}")
    print(f"Epochs:              {epochs}")
    print(f"Batch size:          {batch_size} (effective: {batch_size * accum_steps})")
    print(f"Learning rate:       {lr}")
    print(f"Warmup ratio:        {warmup_ratio}")
    print(f"Focal loss gamma:    {focal_gamma}")
    print(f"Label smoothing:     {label_smoothing}")
    print(f"Early stop patience: {patience}")
    print(f"Model: {num_layers} layers, d_ff={d_ff}, dropout={dropout}, CRF={use_crf}")
    print(f"KD Temp: {temperature}, Logit KD: {kd_weight}, Hidden KD: {hidden_kd_weight}, Attn KD: {attention_kd_weight}")
    print(f"{'='*60}\n")

    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    # Optional: mix in pseudo-labeled data (output of pseudo_label.py).
    # We wrap the labeled DataLoader's dataset and the pseudo-records in a ConcatDataset,
    # which preserves the existing collation/batch_size/shuffle behavior of train_loader.
    if pseudo_data_path is not None and os.path.exists(pseudo_data_path):
        from torch.utils.data import ConcatDataset, DataLoader, Dataset

        class _PseudoDataset(Dataset):
            def __init__(self, records):
                self.records = records
            def __len__(self):
                return len(self.records)
            def __getitem__(self, i):
                return self.records[i]

        pseudo_records = torch.load(pseudo_data_path, map_location="cpu")
        print(f"Loaded {len(pseudo_records)} pseudo-labeled examples from {pseudo_data_path}.")
        merged = ConcatDataset([train_loader.dataset, _PseudoDataset(pseudo_records)])
        train_loader = DataLoader(
            merged,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=getattr(train_loader, "num_workers", 0),
            collate_fn=train_loader.collate_fn,
        )
        print(f"Merged training set size: {len(merged)}")

    # Compute class weights
    class_weights = compute_class_weights(train_loader, num_classes=5).to(device)

    # Initialize model — bigger architecture, CRF on top
    print("\nInitializing model...")
    model = SkillExtractor(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=5,
        max_len=max_length,
        dropout=dropout,
        use_crf=use_crf
    ).to(device)

    param_count = model.get_num_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"Under 6M budget: {'YES' if param_count < 6_000_000 else 'NO'}\n")

    # Optional warm-start: load a PCA-projected BERT word_embeddings matrix
    # (30522, d_model) into the student's embedding table. The embedding is the
    # majority of the student's params, so a non-random starting point is high
    # leverage — and free at inference (same parameter count, same checkpoint
    # format). Generated by init_embedding_pca.py.
    if init_embedding_path is not None and os.path.exists(init_embedding_path):
        pca_emb = torch.load(init_embedding_path, map_location=device)
        expected_shape = model.embedding.weight.shape
        if pca_emb.shape != expected_shape:
            raise ValueError(
                f"PCA embedding shape {tuple(pca_emb.shape)} doesn't match student "
                f"embedding shape {tuple(expected_shape)}."
            )
        with torch.no_grad():
            model.embedding.weight.copy_(pca_emb)
        print(f"Initialized student embedding from {init_embedding_path} "
              f"(std={float(pca_emb.std()):.3f})\n")
    elif init_embedding_path is not None:
        print(f"WARNING: init_embedding_path={init_embedding_path} not found — "
              f"falling back to random init.\n")

    # ---- Load Teacher Model ----
    print("\nLoading Teacher Model for Distillation...")
    # `attn_implementation="eager"` is required to expose attention weights for
    # attention-map distillation; the default SDPA kernel returns None for attentions.
    teacher_model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", num_labels=5, attn_implementation="eager"
    ).to(device)
    if os.path.exists(teacher_model_path):
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        print(f"Teacher loaded from {teacher_model_path}")
    else:
        print(f"WARNING: Teacher checkpoint not found at {teacher_model_path}! Using untrained teacher!")
    teacher_model.eval()
    # Force config flags on — some HF versions only honor these at the config level,
    # ignoring runtime kwargs. Setting both guarantees teacher_outputs.attentions /
    # .hidden_states are populated.
    teacher_model.config.output_hidden_states = True
    teacher_model.config.output_attentions = True
    for param in teacher_model.parameters():
        param.requires_grad = False

    # ---- Loss Functions ----
    focal_criterion = FocalLoss(
        alpha=class_weights, gamma=focal_gamma,
        ignore_index=-100, label_smoothing=label_smoothing
    )
    val_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    # ---- Hidden-state KD projection ----
    # Projects student hidden states (d_model=128) up to the teacher's hidden size
    # so we can compute MSE against DistilBERT's per-layer hidden states.
    # One shared linear projection across all matched layers keeps the param cost low
    # and is only used during training (not exported with the student checkpoint).
    hidden_projector = nn.Linear(128, teacher_hidden_size).to(device) if hidden_kd_weight > 0 else None

    # ---- Optimizer ----
    trainable_params = list(model.parameters())
    if hidden_projector is not None:
        trainable_params += list(hidden_projector.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # ---- Scheduler ----
    num_training_steps = epochs * len(train_loader) // accum_steps
    warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, num_training_steps)

    print(f"Total optimizer steps: {num_training_steps}")
    print(f"Warmup steps:          {warmup_steps}\n")

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = 0.0
    epochs_without_improvement = 0

    print("Starting training...\n")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            need_hidden = hidden_projector is not None
            need_attn = attention_kd_weight > 0
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # Teacher Forward Pass — request hidden states / attention for layer-wise distillation
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=need_hidden,
                        output_attentions=need_attn,
                    )
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states if need_hidden else None
                    teacher_attentions = teacher_outputs.attentions if need_attn else None

                # Student Forward Pass
                if need_hidden or need_attn:
                    student_logits, student_extras = model(
                        input_ids, attention_mask,
                        return_hidden_states=need_hidden,
                        return_attentions=need_attn,
                    )
                    student_hidden_states = student_extras.get("hidden_states")
                    student_attentions = student_extras.get("attentions")
                else:
                    student_logits = model(input_ids, attention_mask)
                    student_hidden_states = None
                    student_attentions = None

                # --- Active Tokens Mask ---
                active_mask = attention_mask.view(-1) == 1
                flat_student_logits = student_logits.view(-1, model.classifier.out_features)
                flat_teacher_logits = teacher_logits.view(-1, teacher_model.num_labels)
                flat_labels = labels.view(-1)

                # --- Focal Loss (Hard Loss) ---
                focal_loss = focal_criterion(flat_student_logits, flat_labels)

                # --- CRF Loss (Sequence-level Hard Loss) ---
                if use_crf:
                    crf_labels = labels.clone()
                    crf_labels[crf_labels == -100] = 0
                    crf_loss = model.crf(student_logits.float(), crf_labels, attention_mask.float())
                    hard_loss = crf_loss_weight * crf_loss + focal_loss_weight * focal_loss
                else:
                    hard_loss = focal_loss

                # --- KL Divergence (Soft Loss / Knowledge Distillation) ---
                active_student_logits = flat_student_logits[active_mask]
                active_teacher_logits = flat_teacher_logits[active_mask]
                
                # Temperature scaling
                scaled_student_log_probs = F.log_softmax(active_student_logits / temperature, dim=-1)
                scaled_teacher_probs = F.softmax(active_teacher_logits / temperature, dim=-1)
                
                # KLDiv expects inputs as log-probabilities, and targets as probabilities
                kd_loss = kl_div_loss(scaled_student_log_probs, scaled_teacher_probs) * (temperature ** 2)

                # --- Hidden-state KD (TinyBERT-style layer matching) ---
                # Student exposes `num_layers + 1` hidden states (embedding + each block).
                # BERT-base teacher exposes 13 hidden states (embedding + 12 layers). With
                # a 6-layer student, naive zip would only see the lower half of the teacher
                # and ignore the deepest layers (where the most useful semantic features
                # live). Subsample the teacher to len(student) using uniformly-spaced
                # indices — student layer i ↔ teacher layer round(i * (T-1)/(S-1)).
                # This is the TinyBERT / MobileBERT standard recipe.
                #
                # Computed with autocast DISABLED — BERT-base activations are large enough
                # that the squared MSE overflows fp16, producing inf gradients that
                # GradScaler then skips every step on (silent training failure).
                if hidden_projector is not None and student_hidden_states is not None:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        valid_mask = attention_mask.unsqueeze(-1).float()
                        num_valid = valid_mask.sum().clamp(min=1.0)

                        n_s = len(student_hidden_states)
                        n_t = len(teacher_hidden_states)
                        if n_t > n_s:
                            step_size = (n_t - 1) / (n_s - 1) if n_s > 1 else 1
                            teacher_layers = [teacher_hidden_states[round(i * step_size)] for i in range(n_s)]
                        else:
                            teacher_layers = list(teacher_hidden_states)

                        hidden_loss = 0.0
                        matched = 0
                        for s_h, t_h in zip(student_hidden_states, teacher_layers):
                            s_proj = hidden_projector(s_h.float())
                            diff = (s_proj - t_h.float()) * valid_mask
                            hidden_loss = hidden_loss + (diff.pow(2).sum() / (num_valid * teacher_hidden_size))
                            matched += 1
                        hidden_loss = hidden_loss / max(1, matched)
                else:
                    hidden_loss = torch.tensor(0.0, device=device)

                # --- Attention-map KD (TinyBERT-style) ---
                # Student: 4 heads × num_layers. Teacher (BERT-base): 12 heads × 12 layers.
                # We resolve the head-count mismatch by averaging attention probs over heads
                # on both sides (TinyBERT's standard recipe), then matching layer-by-layer
                # using the same subsampling indices as hidden KD. Attention maps are
                # probabilities (rows sum to 1), so MSE values are small — needs a larger
                # weight than hidden KD to contribute meaningfully (see default=10.0).
                if need_attn and student_attentions is not None and teacher_attentions is not None:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        # Mask: only score attention on real-token positions in both axes.
                        # (B, 1, 1, L) * (B, 1, L, 1) -> (B, 1, L, L)
                        am = attention_mask.float()
                        pair_mask = am.unsqueeze(1).unsqueeze(2) * am.unsqueeze(1).unsqueeze(3)
                        num_valid_pairs = pair_mask.sum().clamp(min=1.0)

                        n_s_a = len(student_attentions)
                        n_t_a = len(teacher_attentions)
                        if n_t_a > n_s_a:
                            step_a = (n_t_a - 1) / (n_s_a - 1) if n_s_a > 1 else 1
                            teacher_attn_layers = [teacher_attentions[round(i * step_a)] for i in range(n_s_a)]
                        else:
                            teacher_attn_layers = list(teacher_attentions)

                        attn_loss = 0.0
                        matched_a = 0
                        for s_a, t_a in zip(student_attentions, teacher_attn_layers):
                            # Average over heads → (B, 1, L, L) on both sides.
                            s_a_mean = s_a.float().mean(dim=1, keepdim=True)
                            t_a_mean = t_a.float().mean(dim=1, keepdim=True)
                            diff = (s_a_mean - t_a_mean) * pair_mask
                            attn_loss = attn_loss + diff.pow(2).sum() / num_valid_pairs
                            matched_a += 1
                        attn_loss = attn_loss / max(1, matched_a)
                else:
                    attn_loss = torch.tensor(0.0, device=device)

                # Total Loss
                loss = (
                    (1.0 - kd_weight) * hard_loss
                    + kd_weight * kd_loss
                    + hidden_kd_weight * hidden_loss
                    + attention_kd_weight * attn_loss
                )
                loss = loss / accum_steps

            # Surface silent failures: if any single batch produces non-finite loss,
            # crash now rather than spending an hour wondering why val F1 is frozen.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch+1} step {step}: "
                    f"loss={loss.item()}, hard={hard_loss.item():.3f}, "
                    f"kd={kd_loss.item():.3f}, hidden={hidden_loss.item():.3f}, "
                    f"attn={attn_loss.item():.3f}"
                )

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        val_results = evaluate_model(model, val_loader, device, val_criterion)
        print(f"\n  Epoch {epoch+1:>2}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_results['loss']:.4f} | "
              f"Val F1: {val_results['f1']:.4f} | "
              f"Val Prec: {val_results['precision']:.4f} | "
              f"Val Rec: {val_results['recall']:.4f}")

        # ---- Save best + Early stopping ----
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            epochs_without_improvement = 0
            print(f"  >>> New best F1: {best_f1:.4f} — saving checkpoint")
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
        else:
            epochs_without_improvement += 1
            print(f"  --- No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs.")
            break

        print()

    # ---- Final Test Evaluation ----
    print(f"\n{'='*60}")
    print("  FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt", map_location=device))

    print("\nBenchmarking inference on CPU...")
    model.to("cpu")
    cpu_test_results = evaluate_model(model, test_loader, torch.device("cpu"))
    print_detailed_metrics(cpu_test_results)

    print(f"\nBest Validation F1: {best_f1:.4f}")
    print(f"Checkpoint saved at: {save_dir}/best_model.pt")


if __name__ == "__main__":
    train(
        epochs=50,
        batch_size=32,
        lr=5e-4,
        max_length=128,
        warmup_ratio=0.1,
        accum_steps=2,
        patience=10,
        focal_gamma=2.0,
        label_smoothing=0.05,
        num_layers=6,
        d_ff=768,
        dropout=0.2,
        use_crf=True,
        crf_loss_weight=1.0,
        focal_loss_weight=0.5,
        pseudo_data_path="data/pseudo_v2.pt",
        init_embedding_path="data/bert_embedding_pca128_teacher.pt",
    )
