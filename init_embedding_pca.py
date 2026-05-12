"""Project BERT's pretrained word-embedding table (30522 x 768) down to the
student's d_model (128) via PCA, and save the result as a 30522 x 128 tensor.

Loaded by train.py to initialize the student's `embedding.weight` instead of
random init. This is the TinyBERT/MobileBERT-style cheap warm-start — the
embedding is ~70% of the student's params, so giving it a non-random starting
point is high leverage.

Run once:
  python init_embedding_pca.py
"""
import argparse
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import BertModel, BertForTokenClassification


def load_word_embeddings(model_name, finetuned_ckpt=None, num_labels=5):
    """Return the (vocab, hidden) word-embedding matrix from either the vanilla
    pretrained model or a fine-tuned-on-our-task checkpoint."""
    if finetuned_ckpt and os.path.exists(finetuned_ckpt):
        print(f"Loading FINE-TUNED teacher from {finetuned_ckpt} (architecture: {model_name})…")
        model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        state = torch.load(finetuned_ckpt, map_location="cpu")
        model.load_state_dict(state)
        return model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
    print(f"Loading vanilla pretrained {model_name}…")
    bert = BertModel.from_pretrained(model_name)
    return bert.embeddings.word_embeddings.weight.detach().cpu().numpy()


def main(
    model_name="bert-base-uncased",
    finetuned_ckpt=None,
    target_dim=128,
    output_path="data/bert_embedding_pca128.pt",
):
    emb = load_word_embeddings(model_name, finetuned_ckpt)
    print(f"BERT word_embeddings shape: {emb.shape}")
    assert emb.shape[1] >= target_dim, (
        f"Cannot project {emb.shape[1]}-dim embeddings down to {target_dim}."
    )

    print(f"Fitting PCA to {target_dim} components…")
    pca = PCA(n_components=target_dim, random_state=0)
    emb_pca = pca.fit_transform(emb)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"Explained variance retained: {explained * 100:.2f}%")
    print(f"PCA-projected shape: {emb_pca.shape}")

    tensor = torch.from_numpy(emb_pca.astype(np.float32))
    raw_std = float(tensor.std())
    print(f"Raw per-element std after PCA: {raw_std:.4f}")

    # Rescale to match nn.Embedding's default init scale (~1.0). PCA's low-variance
    # components scale the projected vectors down sharply; without rescaling, the
    # student's encoder sees tiny inputs and the downstream LayerNorms behave
    # very differently than during random-init training.
    tensor = tensor / raw_std
    print(f"Rescaled per-element std: {float(tensor.std()):.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(tensor, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--finetuned_ckpt", default=None,
                        help="Optional path to a fine-tuned teacher checkpoint. "
                             "If given, PCA is run on the fine-tuned embedding instead "
                             "of the vanilla pretrained one.")
    parser.add_argument("--target_dim", type=int, default=128)
    parser.add_argument("--output_path", default="data/bert_embedding_pca128.pt")
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        finetuned_ckpt=args.finetuned_ckpt,
        target_dim=args.target_dim,
        output_path=args.output_path,
    )
