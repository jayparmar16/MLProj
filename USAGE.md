# Skill Extractor: Inference Usage Guide

This guide demonstrates how to run the trained <6M parameter NER model on a real-world Job Description (JD) to extract **Skill** and **Knowledge** spans, following the ESCO / SkillSpan taxonomy.

## Label categories at a glance

| Label | Meaning | Examples |
|---|---|---|
| `Skill` | Any ability or competence — technical or soft. | Python, Docker, Kubernetes, communication, leadership, mentoring, English |
| `Knowledge` | A body of theoretical or academic understanding. | linear algebra, statistics, consumer psychology, project management methodology |

> The `Skill` class was historically labelled `Tech` in this codebase. That name was misleading: it suggested programming-only, but SkillSpan's underlying definition covers *any* competence. The labels were renamed for clarity. The model's class indices (0..4) are unchanged — only the human-readable names moved.

## How to run

```bash
python infer.py
```

*The script automatically detects a CUDA GPU and uses it if available. For a single short JD on CPU, expect ~30 ms inference latency.*

## Sample input

The script processes raw text. The sample JD currently configured in `infer.py`:

> "We are looking for a Senior Backend Engineer to join our core infrastructure team. You must have strong problem-solving skills and excellent communication abilities to work with cross-functional teams. Experience with Python, Django, and PostgreSQL is required. Familiarity with containerization technologies like Docker and Kubernetes is a huge plus. You should also be capable of mentoring junior developers and leading architectural discussions."

## Sample output

```text
Loaded checkpoint from ./checkpoints/best_model.pt

--- Job Description Excerpt ---
We are looking for a Senior Backend Engineer to join our core infrastructure team. ...
-------------------------------

Extracted Skills:
  - leading architectural discussions
  - infrastructure
  - Django
  - Docker
  - problem
  - Kubernetes
  - mentoring junior developers
  - Python
  - communication abilities
  - work with cross
  - solving skills

Extracted Knowledge:
  - containerization
```
