import torch
from transformers import AutoTokenizer
from model import SkillExtractor
import time

def run_benchmark():
    # 100-word job description
    text = (
        "We are actively seeking a highly skilled and motivated Senior Software Engineer "
        "to join our dynamic backend development team. The ideal candidate will have extensive "
        "experience designing, building, and maintaining scalable web applications and microservices. "
        "You must possess a deep understanding of Python, Django, and RESTful API architecture. "
        "Proficiency in database design and optimization using PostgreSQL and Redis is absolutely "
        "essential for this role. Furthermore, hands-on experience with cloud infrastructure, "
        "specifically AWS services like EC2, S3, and Lambda, along with containerization tools "
        "such as Docker and Kubernetes, is required. Strong communication skills are a must, "
        "as you will collaborate closely with cross-functional teams including frontend developers, "
        "product managers, and UX designers. Mentoring junior developers and leading code reviews "
        "are also key responsibilities. A bachelor's degree in Computer Science is preferred."
    )
    
    # Pre-tokenize to just measure model inference separately if we want, 
    # but let's measure end-to-end (tokenizer + model + decode)
    
    print(f"Word count: {len(text.split())} words")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SkillExtractor(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=6,
        d_ff=768,
        num_classes=5,
        max_len=256,
        use_crf=True
    )
    model.eval()
    
    def benchmark_device(device_name, device):
        model.to(device)
        
        # Warmup
        for _ in range(10):
            encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                if hasattr(model, 'use_crf'):
                    preds = model.crf.decode(logits.float(), attention_mask.float())
                else:
                    preds = torch.argmax(logits, dim=-1)
        
        # Benchmark
        iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                if hasattr(model, 'use_crf'):
                    preds = model.crf.decode(logits.float(), attention_mask.float())
                else:
                    preds = torch.argmax(logits, dim=-1)
                    
            # Synchronize if CUDA to get accurate timings
            if device.type == "cuda":
                torch.cuda.synchronize()
                
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        print(f"[{device_name.upper()}] Average End-to-End Inference Time (100 words): {avg_time_ms:.2f} ms")

    benchmark_device("CPU", torch.device("cpu"))
    
    if torch.cuda.is_available():
        benchmark_device("CUDA (RTX 3080)", torch.device("cuda"))

if __name__ == "__main__":
    run_benchmark()
