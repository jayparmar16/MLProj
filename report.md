# Cross-Domain Skill Extraction - Project Report

## Abstract Summary
This project aimed to design and train a custom, ultra-lightweight Encoder-only Transformer (~4.7M parameters) for Named Entity Recognition (NER) on the SkillSpan dataset. The primary goal was to explore the mechanics of token classification by building the network from scratch while leveraging a robust, pre-trained tokenizer (BERT-base WordPiece). The resulting model maps input tokens to 5 target classes (`O`, `B-Tech`, `I-Tech`, `B-Knowledge`, `I-Knowledge`) and runs efficiently on CPU environments.

## Architecture and Budget
To fit the model under the strict 6M parameter budget while maintaining enough capacity for cross-domain representations, the following design decisions were made:

- **Tokenizer**: BERT-base WordPiece tokenizer (30,522 vocab size). Borrowed, frozen, and contributing exactly `0` trainable parameters.
- **Embedding Layer**: The most expensive part of the network, weighing in at `30,522 × 128 = ~3.9M parameters`.
- **Transformer Encoder**:
  - 4 Layers
  - `d_model` = 128
  - `num_heads` = 4
  - `d_ff` = 512
  - Pre-LayerNorm architecture for training stability.
- **Classification Head**: A simple linear projection from `128` to `5` classes.
- **Total Trainable Parameters**: `4,700,805` (Comfortably under the 6M budget).

## Subword Label Alignment
One of the most complex components of NER with subword tokenizers is label alignment. The WordPiece tokenizer routinely splits single words (e.g., "communication") into multiple tokens (`["commun", "##ication"]`).

In our pipeline, we map the provided tags from both `tags_skill` (Tech) and `tags_knowledge` (Knowledge). We assign the true label to the **first subword token** of a given word, and we assign a label of `-100` to all subsequent subword pieces, padding tokens, and special tokens (`[CLS]`, `[SEP]`). The `CrossEntropyLoss` function intrinsically ignores the `-100` index, ensuring that the model is only penalized for misclassifying the leading subwords, preventing an artificial inflation of evaluation metrics on easy-to-predict word suffix fragments.

## Training Optimizations
The training pipeline was written to automatically detect the availability of hardware accelerators (CUDA/MPS/CPU). When a CUDA GPU is present, it uses `torch.autocast` to perform FP16 mixed-precision training. We utilized the `AdamW` optimizer combined with a `CosineAnnealingLR` scheduler and gradient clipping (`max_norm=1.0`) to avoid catastrophic gradient explosions, a common failure mode when training small transformers from scratch.

## Limitations and Failure Modes
Given the stringent constraints of this project, several tradeoffs had to be acknowledged:
1. **Low Precision on Soft Skills**: As anticipated in the uncertainties, mapping multi-word contextual phrases (Knowledge/Soft skills) reliably proved difficult for an untrained model with only 4.7M parameters. The baseline metrics reflect near-zero initial recall on `B-Knowledge` and `I-Knowledge`, demonstrating the limit of small-capacity models without pre-trained attention heads.
2. **Data Imbalance**: The vast majority of tokens evaluate to `O` (outside). In our benchmark run, 85% of predictions gravitated toward `O`, making it difficult for the model to initially learn the nuanced boundaries of technical skills without techniques like focal loss or rigorous hyperparameter tuning over extensive epochs.
3. **Training Time Required**: To reach optimal metrics, this scratch-built model would require substantially more epochs than fine-tuning a pre-trained BERT architecture.

## Conclusion
Despite performance challenges resulting from the parameter budget constraint, the project serves as a compelling exploration of from-scratch deep learning. We demonstrated successful separation of the tokenizer from the core model architecture, implemented standard transformer components, correctly solved the subword alignment edge cases, and constructed an inference loop highly optimized for CPU latency (~10s inference over the entire SkillSpan test set of >3500 examples).
