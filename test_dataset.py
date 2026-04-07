from datasets import load_dataset
dataset = load_dataset("jjzha/skillspan")
for item in dataset["train"]:
    for s in item["tags_skill"]:
        if s != 'O':
            print(repr(s))
            break
    else:
        continue
    break
