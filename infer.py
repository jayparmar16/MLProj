import torch
from transformers import AutoTokenizer
from model import SkillExtractor
from dataset import ID_TO_LABEL
from skill_taxonomy import classify_span
import os

def load_model(checkpoint_path="./checkpoints/best_model.pt", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Same architecture parameters as the v2 training
    model = SkillExtractor(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=6,
        d_ff=768,
        num_classes=5,
        max_len=128, # Adjust if we pass longer texts, but for now we'll chunk or truncate
        dropout=0.2,
        use_crf=True
    )
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained weights.")
        
    model.to(device)
    model.eval()
    return model, tokenizer

def extract_skills(text, model, tokenizer, device="cpu"):
    # Tokenize. We request both offset_mapping (subword char ranges) and
    # word_ids (the parent-word index for each subword), because the model
    # was trained to predict labels on the FIRST subword of each word and
    # to treat continuation subwords as ignored (-100). At inference we must
    # mirror that: ignore continuation-subword predictions, but still extend
    # the current span's character range to include them so we don't emit
    # fragments like "Ku" instead of "Kubernetes".
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    word_ids = encoding.word_ids(batch_index=0)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        if hasattr(model, "use_crf") and model.use_crf:
            preds = model.crf.decode(logits.float(), attention_mask.float())
        else:
            preds = torch.argmax(logits, dim=-1)

    preds = preds[0].cpu().numpy()

    extracted_skills = {"Skill": [], "Knowledge": []}
    current_skill_type = None
    current_span_start = -1
    current_span_end = -1
    previous_word_idx = None

    def flush_current():
        nonlocal current_skill_type, current_span_start, current_span_end
        if current_skill_type and current_span_start != -1:
            skill_text = text[current_span_start:current_span_end].strip()
            if skill_text:
                # Post-processing override: when the extracted phrase has an
                # unambiguous match in the curated taxonomy, trust the
                # taxonomy over the model's category prediction. Fixes the
                # known Python-as-Knowledge / Kubernetes-as-Knowledge errors
                # that come from inconsistent SkillSpan gold labels.
                override = classify_span(skill_text)
                target = override or current_skill_type
                extracted_skills[target].append(skill_text)
        current_skill_type = None
        current_span_start = -1
        current_span_end = -1

    for idx in range(len(preds)):
        if attention_mask[0, idx] == 0:
            continue
        word_idx = word_ids[idx]
        if word_idx is None:
            # Special tokens ([CLS], [SEP], [PAD]) — no parent word.
            continue

        start, end = offset_mapping[idx]

        # Continuation subword: do not read its prediction — the model was
        # never trained to produce a meaningful one. Just extend the current
        # span's char range so we output the full word, not the first piece.
        if word_idx == previous_word_idx:
            if current_skill_type is not None:
                current_span_end = max(current_span_end, end)
            continue
        previous_word_idx = word_idx

        label = ID_TO_LABEL[preds[idx]]

        if label.startswith("B-"):
            flush_current()
            current_skill_type = "Skill" if "Skill" in label else "Knowledge"
            current_span_start = start
            current_span_end = end
        elif label.startswith("I-") and current_skill_type:
            current_span_end = max(current_span_end, end)
        else:  # "O" or stray I- with no open span
            flush_current()

    flush_current()
    return extracted_skills

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device=device)
    
    # Real-world Software Engineering JD excerpt
    jd_excerpt = (
        "We are looking for a Senior Backend Engineer to join our core infrastructure team. "
        "You must have strong problem-solving skills and excellent communication abilities to work with cross-functional teams. "
        "Experience with Python, Django, and PostgreSQL is required. "
        "Familiarity with containerization technologies like Docker and Kubernetes is a huge plus. "
        "You should also be capable of mentoring junior developers and leading architectural discussions."
    )
    
    print("\n--- Job Description Excerpt ---")
    print(jd_excerpt)
    print("-------------------------------\n")
    
    # Since max_length is 128, we'll process sentence by sentence or just chunk it.
    # The excerpt above is short enough to fit in 128 tokens.
    
    skills = extract_skills(jd_excerpt, model, tokenizer, device=device)
    
    # Per SkillSpan / ESCO: "Skill" = any competence (technical or soft);
    # "Knowledge" = a theoretical / academic body. The earlier name "Tech" for
    # the Skill class was misleading — it suggested programming-only.
    print("Extracted Skills:")
    for s in set(skills["Skill"]):
        print(f"  - {s}")

    print("\nExtracted Knowledge:")
    for s in set(skills["Knowledge"]):
        print(f"  - {s}")
