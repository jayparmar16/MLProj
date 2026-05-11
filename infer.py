import torch
from transformers import AutoTokenizer
from model import SkillExtractor
from dataset import ID_TO_LABEL
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
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True, # Helps map back to original text
        add_special_tokens=True
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().numpy()
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        if hasattr(model, 'use_crf') and model.use_crf:
            preds = model.crf.decode(logits.float(), attention_mask.float())
        else:
            preds = torch.argmax(logits, dim=-1)
            
    preds = preds[0].cpu().numpy()
    
    # Reconstruct spans using actual string slicing
    extracted_skills = {"Tech": [], "Knowledge": []}
    
    current_skill_type = None
    current_span_start = -1
    current_span_end = -1
    
    for idx, pred in enumerate(preds):
        if attention_mask[0, idx] == 0:
            continue
            
        label = ID_TO_LABEL[pred]
        start, end = offset_mapping[idx]
        
        # Skip special tokens
        if start == 0 and end == 0:
            continue
            
        if label.startswith("B-"):
            # Save previous skill
            if current_skill_type and current_span_start != -1:
                skill_text = text[current_span_start:current_span_end].strip()
                if skill_text:
                    extracted_skills[current_skill_type].append(skill_text)
                
            current_skill_type = "Tech" if "Tech" in label else "Knowledge"
            current_span_start = start
            current_span_end = end
            
        elif label.startswith("I-") and current_skill_type:
            # Extend the span
            current_span_end = max(current_span_end, end)
                
        elif label == "O":
            if current_skill_type and current_span_start != -1:
                skill_text = text[current_span_start:current_span_end].strip()
                if skill_text:
                    extracted_skills[current_skill_type].append(skill_text)
            current_skill_type = None
            current_span_start = -1
            
    # Add the last one if present
    if current_skill_type and current_span_start != -1:
        skill_text = text[current_span_start:current_span_end].strip()
        if skill_text:
            extracted_skills[current_skill_type].append(skill_text)
        
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
    
    print("Extracted Technical Skills:")
    for s in set(skills["Tech"]):
        print(f"  - {s}")
        
    print("\nExtracted Soft/Knowledge Skills:")
    for s in set(skills["Knowledge"]):
        print(f"  - {s}")
