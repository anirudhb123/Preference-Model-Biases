from absl import app
from absl import flags
from peft import PeftModel

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import jsonl_utils
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import logging


# Flags for normal evaluation on a given input file.
_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing query, base_response, and perturbed_response fields."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file for scored responses."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2", "Model name."
)

_ADAPTER_NAME = flags.DEFINE_string(
    "adapter_name", "", "Adapter name."
)

# Flags for training data evaluation (sanity check)
_TRAIN_EVAL = flags.DEFINE_boolean(
    "train_eval", False, "If true, perform training data evaluation (chosen_response vs. rejected_response)."
)
_TRAINING_DATA_PATH = flags.DEFINE_string(
    "training_data_path", "", "Path to the training data file with query, chosen_response, and rejected_response fields."
)


def score_chat(chat, model, tokenizer):
    device = "cuda:0"
    # Create the message template (assuming the tokenizer provides an apply_chat_template method)
    message_template = tokenizer.apply_chat_template(chat, tokenize=False)
    tokens = tokenizer(message_template, return_tensors="pt").to(device)
    with torch.no_grad():
        reward_tensor = model(**tokens).logits[0][0].item()
    return reward_tensor


def main(unused_argv) -> None:
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME.value)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_NAME.value,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        num_labels=1,
        # load_in_8bit=True
    )

    # Swap in the specified adapter weights
    print(f"Loading Model: {_MODEL_NAME.value}")
    if _ADAPTER_NAME.value:
        print(f"Loading PEFT adapter: {_ADAPTER_NAME.value}")
        model = PeftModel.from_pretrained(
            base_model,
            _ADAPTER_NAME.value
        )
    else:
        model = base_model

    # If an input file is provided, process it as usual.
    if _INPUT_PATH.value:
        input_data = jsonl_utils.read(_INPUT_PATH.value)
        formatted_output = []
        print(f"Processing {len(input_data)} examples from {_INPUT_PATH.value}...")
        for example in tqdm.tqdm(input_data, desc="Scoring examples"):
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
        print(f"Scores written to {_OUTPUT_PATH.value}")

    # If training evaluation is requested, process up to 399 training examples.
    if _TRAIN_EVAL.value and _TRAINING_DATA_PATH.value:
        training_data = jsonl_utils.read(_TRAINING_DATA_PATH.value)
        num_examples = min(len(training_data), 399)
        training_data = training_data[:num_examples]
        print(f"Evaluating training data on {num_examples} examples...")

        correct = 0
        detailed_results = []
        for idx, example in enumerate(tqdm.tqdm(training_data, desc="Evaluating training examples")):
            query = example['query']
            chosen_response = example['chosen']
            rejected_response = example['rejected']

            chat_chosen = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": chosen_response}
            ]
            chat_rejected = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": rejected_response}
            ]

            score_chosen = score_chat(chat_chosen, model, tokenizer)
            score_rejected = score_chat(chat_rejected, model, tokenizer)

            if score_chosen > score_rejected:
                correct += 1

            #detailed_results.append({
            #    'query': query,
            #    'chosen_response': chosen_response,
            #    'chosen_score': score_chosen,
            #    'rejected_response': rejected_response,
            #    'rejected_score': score_rejected,
            #    'chosen_higher': score_chosen > score_rejected,
            #})

            # Clear GPU memory every 50 examples
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        accuracy = (correct / num_examples) * 100
        print(f"Training evaluation: Chosen response scored higher than rejected response in {correct}/{num_examples} examples ({accuracy:.2f}%).")

        # Optionally, you can write detailed_results to a file for further inspection.
        # jsonl_utils.write("training_evaluation_results.jsonl", detailed_results)


if __name__ == "__main__":
    app.run(main)
