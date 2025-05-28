r"""Generate prefernce model scores for a given set of queries and model responses.

Example usage:

INPUT_PATH=data/all_data_perturbed_structure.jsonl
OUTPUT_PATH=data/structure_scores.jsonl
MODEL_NAME=Skywork/Skywork-Reward-Gemma-2-27B-v0.2
python3 main/run_preference_model.py \
--model_name=${MODEL_NAME} \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

from absl import app
from absl import flags

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import jsonl_utils
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2", "Model name."
)


def score_chat(chat, model, tokenizer):
    device = "cuda:0"
    message_template = tokenizer.apply_chat_template(chat, tokenize=False)    
    tokens = tokenizer(message_template, return_tensors="pt").to(device)
    with torch.no_grad():
        reward_tensor = model(**tokens).logits[0][0].item()
    return reward_tensor


def main(unused_argv) -> None:
    input_data = jsonl_utils.read(_INPUT_PATH.value)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME.value)
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_NAME.value,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        num_labels=1,
    )

    formatted_output = []
    print(len(input_data))
    for example in tqdm.tqdm(input_data):
        query = example['query']
        response_base = example['base_response']
        response_perturbed = example['perturbed_response']

        chat_base = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_base}
        ]
        chat_perturbed = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_perturbed}
        ]

        score_base = score_chat(chat_base, model, tokenizer)
        score_perturbed = score_chat(chat_perturbed, model, tokenizer)

        formatted_dict = {
            'query': query,
            'base_response': response_base,
            'base_score': score_base,
            'perturbed_response': response_perturbed,
            'perturbed_score': score_perturbed,
        }

        formatted_output.append(formatted_dict)

    jsonl_utils.write(_OUTPUT_PATH.value, formatted_output)



if __name__ == "__main__":
  app.run(main)
