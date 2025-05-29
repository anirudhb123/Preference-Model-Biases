import json
import os

# mapping from bias name to the key in each record
BIAS_KEYS = {
    'length':     'longer_response',
    'structure':  'more_structured_response',
    'jargon':     'more_jargony_response',
    'vagueness':  'more_vague_response',
    'sycophancy': 'more_sycophantic_response',
}

def process_file(path, bias, out_fh):
    print(f"Processing {path}...")
    with open(path, 'r', encoding='utf-8') as fh:
        records = json.load(fh)
    for rec in records:
        human_label = rec.get('most_frequent_preference').lower()

        perturbed = rec.get(BIAS_KEYS[bias])
        if not perturbed:
            continue
        combined = {
            'bias': bias,
            'query': rec.get('query'),
            'response_1': rec.get('response_1'),
            'response_2': rec.get('response_2'),
            'human_preference': human_label,
            'reward_model_preference': rec.get('reward_model_preferred_response'),
            'perturbed (bias-expressing) response': perturbed,
        }
        out_fh.write(json.dumps(combined, ensure_ascii=False) + '\n')

def main():
    out_path = 'combined_human_annotations.jsonl'
    count = 0
    with open(out_path, 'w', encoding='utf-8') as out_fh:
        for bias in BIAS_KEYS:
            folder = bias
            if not os.path.isdir(folder):
                print(f"[!] Missing folder: {folder}")
                continue

            fn = f"{bias}_grouped_human_annotations.json"
            path = os.path.join(folder, fn)
            if not os.path.isfile(path):
                print(f"[!] Missing file: {path}")
                continue

            process_file(path, bias, out_fh)
            count += 1

    print(f"\nDone — processed {count} files, wrote combined records to {out_path}")

if __name__ == '__main__':
    main()
