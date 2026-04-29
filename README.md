# Ultra-Lightweight Skill Extraction Model

This project implements a custom, ultra-lightweight Encoder-only Transformer (~4-6M parameters) from scratch in PyTorch for Named Entity Recognition (NER). It is designed to extract hard ("Tech") and soft ("Knowledge") skills from job descriptions, using the SkillSpan dataset.

## Motivation & Architecture
The model deliberately decouples tokenization from the architecture:
- **Tokenizer**: BERT-base WordPiece tokenizer (30K vocab), borrowed and frozen (0 trainable parameters).
- **Embedding Table**: Trained from scratch (30K × d_model).
- **Transformer Encoder**: 4 layers, implemented from scratch in PyTorch, strictly keeping the parameter count under the 6M threshold.
- **Classification Head**: Linear projection to 5 output classes (`O`, `B-Tech`, `I-Tech`, `B-Knowledge`, `I-Knowledge`).

By constraining the model to <6M parameters and implementing the layers by hand, this project demonstrates that a minimalist LLM approach is viable for cross-domain tasks without relying on massive pre-training or computational resources.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```
   *(Training automatically detects CUDA/MPS/CPU and uses mixed-precision where applicable).*

3. Evaluate:
   ```bash
   python evaluate.py
   ```
