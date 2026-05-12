import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape for multi-head attention
        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # Q @ K^T: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask should be broadcastable to (batch_size, num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)

        # Output: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attn, V)

        # Reshape back to (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(context)

        if return_attention:
            return out, attn
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        # Pre-LN architecture (often more stable for deep networks)
        if return_attention:
            attn_out, attn_probs = self.attention(self.norm1(x), mask, return_attention=True)
        else:
            attn_out = self.attention(self.norm1(x), mask)
            attn_probs = None
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        if return_attention:
            return x, attn_probs
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ---------------------------------------------------------------------------
# CRF Layer — learns valid transition constraints between NER labels
# (e.g., I-Skill can only follow B-Skill or I-Skill, never B-Knowledge)
# ---------------------------------------------------------------------------
class CRF(nn.Module):
    """Linear-chain Conditional Random Field for sequence labeling."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] = score of transitioning FROM tag i TO tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        """
        Compute negative log-likelihood loss.
        emissions: (batch, seq_len, num_tags)  — logits from classifier
        tags:      (batch, seq_len)            — gold labels (NO -100, use 0 for ignored)
        mask:      (batch, seq_len)            — 1 for valid, 0 for padding (float)
        """
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions, mask):
        """Log-sum-exp over all possible tag sequences (partition function)."""
        batch_size, seq_len, num_tags = emissions.shape

        # score: (batch, num_tags) — running log-sum-exp
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            # broadcast_score: (batch, num_tags, 1)
            # transitions:     (num_tags, num_tags) — from j to k
            # emissions_i:     (batch, 1, num_tags)
            broadcast_score = score.unsqueeze(2)
            emissions_i = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + emissions_i
            next_score = torch.logsumexp(next_score, dim=1)  # (batch, num_tags)

            # Only update positions that are valid (not padding)
            mask_i = mask[:, i].unsqueeze(1).bool()
            score = torch.where(mask_i, next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # (batch,)

    def _score_sentence(self, emissions, tags, mask):
        """Score of the gold tag sequence."""
        batch_size, seq_len = tags.shape

        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            mask_i = mask[:, i].bool()
            transition_score = self.transitions[tags[:, i - 1], tags[:, i]]
            emission_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            score += (transition_score + emission_score) * mask_i

        # End transition from the last valid tag
        seq_lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def decode(self, emissions, mask):
        """Viterbi decoding — find the best tag sequence."""
        batch_size, seq_len, num_tags = emissions.shape

        score = self.start_transitions + emissions[:, 0]  # (batch, num_tags)
        history = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)       # (batch, num_tags, 1)
            emissions_i = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)
            next_score = broadcast_score + self.transitions + emissions_i
            next_score, indices = next_score.max(dim=1)  # (batch, num_tags)

            mask_i = mask[:, i].unsqueeze(1).bool()
            score = torch.where(mask_i, next_score, score)
            history.append(indices)

        score = score + self.end_transitions
        _, best_last_tag = score.max(dim=1)  # (batch,)

        # Backtrack
        best_tags = torch.zeros(batch_size, seq_len, dtype=torch.long,
                                device=emissions.device)

        seq_lengths = mask.long().sum(dim=1) - 1

        for b in range(batch_size):
            best_tag = best_last_tag[b].item()
            length = seq_lengths[b].item()
            best_tags[b, length] = best_tag

            for hist_idx in range(length - 1, -1, -1):
                best_tag = history[hist_idx][b, best_tag].item()
                best_tags[b, hist_idx] = best_tag

        return best_tags


class SkillExtractor(nn.Module):
    def __init__(self, vocab_size=30522, d_model=128, num_heads=4, num_layers=4,
                 d_ff=512, num_classes=5, max_len=128, dropout=0.1, use_crf=False):
        super().__init__()

        # 30522 is the standard BERT vocab size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_classes)

    def forward(self, input_ids, attention_mask=None,
                return_hidden_states=False, return_attentions=False):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len), 1 for valid tokens, 0 for padding

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.emb_dropout(x)

        # Create mask for attention: (batch_size, 1, 1, seq_len)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        # Hidden-state distillation: collect per-layer outputs (embedding + each block),
        # mirroring HuggingFace's `output_hidden_states` convention (length = num_layers + 1).
        hidden_states = [x] if return_hidden_states else None
        attentions = [] if return_attentions else None

        for layer in self.layers:
            if return_attentions:
                x, attn = layer(x, mask, return_attention=True)
                attentions.append(attn)
            else:
                x = layer(x, mask)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.final_norm(x)
        logits = self.classifier(x)

        if return_hidden_states or return_attentions:
            extras = {}
            if return_hidden_states:
                extras["hidden_states"] = hidden_states
            if return_attentions:
                extras["attentions"] = attentions
            return logits, extras
        return logits

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Biaffine span head — directly classifies (start, end) token pairs as
# {None, Tech, Knowledge}. Replaces BIO+CRF for span-level NER. The biaffine
# scorer was introduced by Dozat & Manning (2017) for parsing and adapted to
# NER by Yu et al. (2020) — it tends to handle ambiguous span boundaries
# better than tagging schemes because it scores spans as a whole.
# ---------------------------------------------------------------------------
class BiaffineSpanHead(nn.Module):
    def __init__(self, d_model, d_proj=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.start_mlp = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.GELU(), nn.Dropout(dropout)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.GELU(), nn.Dropout(dropout)
        )
        # Biaffine weight: U ∈ (num_classes, d_proj+1, d_proj+1).
        # The +1 absorbs both the bias and the linear (non-bilinear) terms via
        # the standard "append 1" trick: [s; 1]^T U [e; 1].
        self.U = nn.Parameter(torch.empty(num_classes, d_proj + 1, d_proj + 1))
        nn.init.xavier_uniform_(self.U)
        self.num_classes = num_classes

    def forward(self, h):
        # h: (B, L, d_model) -> scores: (B, L, L, num_classes), where dim 1=start, dim 2=end
        s = self.start_mlp(h)
        e = self.end_mlp(h)
        ones_s = torch.ones(*s.shape[:-1], 1, device=s.device, dtype=s.dtype)
        ones_e = torch.ones(*e.shape[:-1], 1, device=e.device, dtype=e.dtype)
        s_aug = torch.cat([s, ones_s], dim=-1)  # (B, L, d_proj+1)
        e_aug = torch.cat([e, ones_e], dim=-1)
        # scores[b, c, i, j] = s_aug[b, i] · U[c] · e_aug[b, j]
        scores = torch.einsum("bxi,cij,byj->bcxy", s_aug, self.U, e_aug)
        return scores.permute(0, 2, 3, 1).contiguous()


class SpanSkillExtractor(nn.Module):
    """Same encoder stack as SkillExtractor, but emits (B, L, L, C) span scores."""
    SPAN_NONE, SPAN_TECH, SPAN_KNOWLEDGE = 0, 1, 2

    def __init__(self, vocab_size=30522, d_model=128, num_heads=4, num_layers=6,
                 d_ff=768, num_span_classes=3, max_len=128, dropout=0.2,
                 d_span_proj=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.span_head = BiaffineSpanHead(d_model, d_span_proj, num_span_classes, dropout)
        self.num_span_classes = num_span_classes

    def forward(self, input_ids, attention_mask=None, return_hidden_states=False):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.emb_dropout(x)
        mask = attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None

        hidden_states = [x] if return_hidden_states else None
        for layer in self.layers:
            x = layer(x, mask)
            if return_hidden_states:
                hidden_states.append(x)
        x = self.final_norm(x)
        span_scores = self.span_head(x)
        if return_hidden_states:
            return span_scores, hidden_states
        return span_scores

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def bio_to_span_labels(bio_labels, max_span_len=20):
    """Convert (B, L) BIO labels to (B, L, L) span class labels.

    BIO ids: 0=O, 1=B-Skill, 2=I-Skill, 3=B-Knowledge, 4=I-Knowledge, -100=ignore.
    Span ids: 0=None, 1=Skill, 2=Knowledge, -100=invalid (i>j, span too long, or padding).
    """
    B, L = bio_labels.shape
    device = bio_labels.device
    span_labels = torch.full((B, L, L), -100, dtype=torch.long, device=device)

    idx = torch.arange(L, device=device)
    upper = idx.unsqueeze(0) <= idx.unsqueeze(1).T  # i <= j
    short = (idx.unsqueeze(0) - idx.unsqueeze(1)) < max_span_len  # j - i < max_span_len
    valid_grid = upper & short  # (L, L)

    span_labels[:, valid_grid] = 0  # default = None

    # Mark padding positions invalid in both axes
    pad_mask = bio_labels == -100  # (B, L)
    for b in range(B):
        bad = pad_mask[b].nonzero(as_tuple=True)[0]
        if bad.numel() > 0:
            span_labels[b, bad, :] = -100
            span_labels[b, :, bad] = -100

    # Decode BIO into spans
    for b in range(B):
        i = 0
        while i < L:
            tag = int(bio_labels[b, i])
            if tag in (1, 3):  # B-Skill or B-Knowledge
                cls = 1 if tag == 1 else 2
                inside = tag + 1
                end = i
                while end + 1 < L and int(bio_labels[b, end + 1]) == inside:
                    end += 1
                if end - i < max_span_len:
                    span_labels[b, i, end] = cls
                i = end + 1
            else:
                i += 1
    return span_labels


def decode_spans(span_scores, attention_mask, max_span_len=20, none_class=0):
    """Greedy non-overlapping decoder.

    span_scores: (B, L, L, C) logits. Per batch item we only enumerate (i, j) with
    i <= j < i + max_span_len AND both within the attended sequence — keeping the
    candidate set ~max_span_len * seq_len instead of L^2. We move to CPU/NumPy
    once up-front to avoid per-cell GPU syncs that made the naive version glacial.

    Returns a list (length B) of lists of (start, end, class_id) with class_id != none_class.
    """
    B, L, _, _ = span_scores.shape
    probs = span_scores.softmax(dim=-1)
    best_score, best_class = probs.max(dim=-1)  # both (B, L, L)

    # Single host transfer; subsequent loops are cheap.
    best_score_np = best_score.detach().cpu().numpy()
    best_class_np = best_class.detach().cpu().numpy()
    seq_lens = attention_mask.sum(dim=1).detach().cpu().numpy()

    out = []
    for b in range(B):
        seq_len = int(seq_lens[b])
        candidates = []
        for i in range(seq_len):
            j_stop = min(seq_len, i + max_span_len)
            for j in range(i, j_stop):
                cls = int(best_class_np[b, i, j])
                if cls == none_class:
                    continue
                candidates.append((float(best_score_np[b, i, j]), i, j, cls))
        candidates.sort(reverse=True)
        taken = [False] * L
        spans = []
        for _, i, j, cls in candidates:
            if any(taken[i:j + 1]):
                continue
            spans.append((i, j, cls))
            for k in range(i, j + 1):
                taken[k] = True
        out.append(spans)
    return out


if __name__ == "__main__":
    # Test both configurations
    for use_crf in [False, True]:
        print(f"\n--- use_crf={use_crf}, 6 layers, d_ff=768, dropout=0.2 ---")
        model = SkillExtractor(
            vocab_size=30522,
            d_model=128,
            num_heads=4,
            num_layers=6,
            d_ff=768,
            num_classes=5,
            dropout=0.2,
            use_crf=use_crf
        )

        param_count = model.get_num_parameters()
        print(f"Total trainable parameters: {param_count:,}")
        print(f"Is within 6M budget: {param_count < 6_000_000}")

        # Test forward pass
        batch_size = 2
        seq_len = 128
        dummy_input = torch.randint(0, 30522, (batch_size, seq_len))
        dummy_mask = torch.ones(batch_size, seq_len)

        logits = model(dummy_input, dummy_mask)
        print(f"Logits shape: {logits.shape}")

        if use_crf:
            dummy_tags = torch.zeros(batch_size, seq_len, dtype=torch.long)
            crf_loss = model.crf(logits.float(), dummy_tags, dummy_mask)
            print(f"CRF loss: {crf_loss.item():.4f}")
            decoded = model.crf.decode(logits.float(), dummy_mask)
            print(f"Decoded shape: {decoded.shape}")
