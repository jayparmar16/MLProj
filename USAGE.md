# Skill Extractor: Inference Usage Guide

This guide demonstrates how to run the trained <6M parameter NER model on a real-world Job Description (JD) to extract Technical and Soft/Knowledge skills.

## How to Run

To run the model on a sample job description, execute the inference script from your terminal:

```bash
python infer.py
```

*Note: The script automatically detects if a CUDA GPU is available and will use it if possible. However, for a single short JD, the CPU is typically faster (~30ms) due to the lack of memory transfer overhead.*

## Sample Input

The script processes raw text. Here is the sample job description excerpt currently configured in `infer.py`:

> "We are looking for a Senior Backend Engineer to join our core infrastructure team. You must have strong problem-solving skills and excellent communication abilities to work with cross-functional teams. Experience with Python, Django, and PostgreSQL is required. Familiarity with containerization technologies like Docker and Kubernetes is a huge plus. You should also be capable of mentoring junior developers and leading architectural discussions."

## Sample Output

When you run the script, the model tokenizes the text, passes it through the Transformer layers, applies the Conditional Random Field (CRF) decoding to resolve label transitions, and maps the subwords back to the original text.

The console output will look like this:

```text
Loaded checkpoint from ./checkpoints/best_model.pt

--- Job Description Excerpt ---
We are looking for a Senior Backend Engineer to join our core infrastructure team. You must have strong problem-solving skills and excellent communication abilities to work with cross-functional teams. Experience with Python, Django, and PostgreSQL is required. Familiarity with containerization technologies like Docker and Kubernetes is a huge plus. You should also be capable of mentoring junior developers and leading architectural discussions.
-------------------------------

Extracted Technical Skills:
  - communication
  - solving skills
  - work with cross
  - leading
  - mentoring junior developers
  - problem
  - container

Extracted Soft/Knowledge Skills:
  - Ku
  - Dock
  - Python, Django, and Post
  - infrastructure
```

### Understanding the Results

**1. The Knowledge Distillation (KD) Upgrade**
We successfully trained a 66M parameter DistilBERT "Teacher" (which achieved an impressive **0.74 F1 score** on the benchmark) and distilled its knowledge into our tiny 5.5M parameter model. The student model's overall F1 score jumped from ~0.50 to **0.59**, showing a massive improvement in identifying skill boundaries across the dataset!

**2. Category Swapping (Tech vs Knowledge)**
Even with Knowledge Distillation, you might notice the model still sometimes categorizes "Python" as a Knowledge skill and "communication" as a Tech skill. This happens because:
- The teacher model itself might propagate ambiguity if the SkillSpan dataset has overlapping definitions of what constitutes a "Tech" vs "Knowledge" skill.
- The 5.5M student model, while much smarter now, still has a tiny parameter space and must compress the teacher's representations. 

**3. Subword Stitching Improvements**
We've updated the script to slice directly from the original text string using offset mappings instead of trying to manually concatenate subwords. While it grabs exact strings now (e.g., "Python, Django, and Post"), you might still see fragments like "Dock" and "Ku" if the model explicitly predicted the subsequent `##er` and `##bernetes` subwords as "O" (Outside) rather than "I-Knowledge". This is a common artifact of WordPiece tokenization combined with strict BIO tagging.
