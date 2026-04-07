import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# The dataset provides 5 distinct tags:
# O: 0
# B-Tech: 1
# I-Tech: 2
# B-Knowledge: 3
# I-Knowledge: 4
ID_TO_LABEL = {0: "O", 1: "B-Tech", 2: "I-Tech", 3: "B-Knowledge", 4: "I-Knowledge"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

class SkillSpanDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        words = item["tokens"]
        # tags_skill contains B/I/O for tech skills
        # tags_knowledge contains B/I/O for knowledge skills
        # We map them to 0-4 labels:
        # O: 0, B-Tech: 1, I-Tech: 2, B-Knowledge: 3, I-Knowledge: 4

        tags_skill = item["tags_skill"]
        tags_knowledge = item["tags_knowledge"]

        labels = []
        for s, k in zip(tags_skill, tags_knowledge):
            # Check for B and I tags, either as strings ("B") or integers if mapping changes
            if s in ("B", 1, "1", "B-Tech"):
                labels.append(1)
            elif s in ("I", 2, "2", "I-Tech"):
                labels.append(2)
            elif k in ("B", 1, "1", "B-Knowledge"):
                labels.append(3)
            elif k in ("I", 2, "2", "I-Knowledge"):
                labels.append(4)
            else:
                labels.append(0)

        # Tokenize the input words
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Subword label alignment
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100 so they are ignored in loss
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the true label
                label = labels[word_idx]
                aligned_labels.append(label)
            else:
                # Subsequent subwords of a word get -100
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        # Convert everything to appropriate tensor formats and remove batch dim
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

def get_dataloaders(batch_size=32, max_length=128):
    """
    Downloads/loads the SkillSpan dataset and creates Train and Validation dataloaders.
    """
    # Load dataset
    dataset = load_dataset("jjzha/skillspan")

    # Using the exact BERT tokenizer with its 30K vocab
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets
    train_dataset = SkillSpanDataset(dataset["train"], tokenizer, max_length=max_length)
    val_dataset = SkillSpanDataset(dataset["validation"], tokenizer, max_length=max_length)
    test_dataset = SkillSpanDataset(dataset["test"], tokenizer, max_length=max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer

if __name__ == "__main__":
    # Test the dataset and label alignment
    print("Testing data pipeline...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(batch_size=2)
    batch = next(iter(train_loader))

    print("Batch Input IDs Shape:", batch["input_ids"].shape)
    print("Batch Labels Shape:", batch["labels"].shape)

    # Verify alignment for a single sample
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print("\nSample Token-Label Alignment:")
    for token, label in zip(tokens[:30], labels[:30]):
        label_str = ID_TO_LABEL.get(label.item(), "IGNORED") if label.item() != -100 else "IGNORED"
        print(f"{token:>15} : {label_str}")
