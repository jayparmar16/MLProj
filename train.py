import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from dataset import get_dataloaders
from model import SkillExtractor
from evaluate import evaluate_model, print_detailed_metrics

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(
    epochs=5,
    batch_size=32,
    lr=3e-4,
    max_length=128,
    save_dir="./checkpoints"
):
    device = get_device()
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        batch_size=batch_size, max_length=max_length
    )

    # Initialize model
    print("Initializing model...")
    model = SkillExtractor(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        num_classes=5,
        max_len=max_length
    ).to(device)

    print(f"Model parameters: {model.get_num_parameters():,}")

    # Loss and Optimizer
    # We ignore index -100 for special tokens and padding
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Scheduler
    num_training_steps = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Mixed precision scaler (only for CUDA)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask)

                # Active elements for loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.classifier.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]

                loss = criterion(active_logits, active_labels)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        print(f"\nEvaluating on validation set...")
        val_results = evaluate_model(model, val_loader, device, criterion)
        print(f"Val Loss: {val_results['loss']:.4f} | Val F1: {val_results['f1']:.4f}")

        # Save best model
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            print(f"New best F1! Saving model to {save_dir}/best_model.pt")
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

    print("\nTraining complete. Evaluating best model on TEST set...")
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt", map_location=device))

    # Benchmarking on CPU strictly
    print("\nBenchmarking Inference on CPU...")
    model.to("cpu")
    cpu_test_results = evaluate_model(model, test_loader, torch.device("cpu"))
    print_detailed_metrics(cpu_test_results)

if __name__ == "__main__":
    train(epochs=1, batch_size=32)
