import json
from datasets import load_dataset
import random

dataset = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")

random.seed(42)  # Ensure reproducibility
sampled_data = random.sample(list(dataset['train']), 10000)

formatted_data = []
for row in sampled_data:
    chosen_content = row['chosen'][1]['content'] if len(row['chosen']) > 1 else ""
    rejected_content = row['rejected'][1]['content'] if len(row['rejected']) > 1 else ""
    content = row['chosen'][0]['content'] if len(row['chosen']) > 0 else ""

    formatted_data.append({
        "query": content,
        "chosen": chosen_content,
        "rejected": rejected_content
    })

output_file = "../data/skywork_training_sample.jsonl"
with open(output_file, 'w') as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")

print(f"Saved formatted data to {output_file}")
