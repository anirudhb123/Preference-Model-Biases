r"""Analyze queries and classify responses into list-related categories.

Example usage:

INPUT_PATH=data/skywork_training_sample.jsonl
OUTPUT_PATH=data/GEMMA_sample_labeled.jsonl
python3 main/list_labeling.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
--model_name=gpt-4
"""

import json
import os
import sys
from absl import app
from absl import flags

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from data_utils import jsonl_utils
import tqdm

_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing queries and responses."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file with labeled list-related classifications."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)

def append_outputs(outputs, output_path):
    """Append the list of outputs to the file in JSONL format."""
    with open(output_path, "a") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

def main(unused_argv) -> None:
    classification_prompt = """
You are a query classifier. Your task is to classify the following query and responses into three categories:
1. Whether the query explicitly or implicitly asks for a list (Yes/No).
2. Whether the chosen response is formatted as a list (Yes/No).
3. Whether the rejected response is formatted as a list (Yes/No).

Here are the query and responses:

Query: [QUERY]

Chosen Response: [CHOSEN]

Rejected Response: [REJECTED]

Provide the answers in the format:
Query Asked for List: [Yes/No]
Chosen is List: [Yes/No]
Rejected is List: [Yes/No]
"""

    examples = jsonl_utils.read(_INPUT_PATH.value)

    if "gpt" in _MODEL_NAME.value:
        model = models.GPT4(model_name=_MODEL_NAME.value)
    elif "gemini" in _MODEL_NAME.value:
        model = models.Gemini(model_name=_MODEL_NAME.value)
    elif "claude" in _MODEL_NAME.value:
        model = models.Claude(model_name=_MODEL_NAME.value)
    elif "jamba" in _MODEL_NAME.value:
        model = models.Jamba(model_name=_MODEL_NAME.value)
    else:
        model = models.TogetherAI(model_name=_MODEL_NAME.value)

    # Create output file if it doesn't exist already.
    if not os.path.exists(_OUTPUT_PATH.value):
        open(_OUTPUT_PATH.value, "w").close()

    batch_outputs = []
    for idx, ex in enumerate(tqdm.tqdm(examples)):
        cur_prompt = (
            classification_prompt
            .replace("[QUERY]", ex.get("query", ""))
            .replace("[CHOSEN]", ex.get("chosen", ""))
            .replace("[REJECTED]", ex.get("rejected", ""))
        )

        generated_output = model.generate(input_text=cur_prompt, max_len=100).strip()

        try:
            parsed_lines = [line.split(": ")[1].strip() for line in generated_output.split("\n") if ": " in line]
            asked_for_list = parsed_lines[0] if len(parsed_lines) > 0 else ""
            chosen_list = parsed_lines[1] if len(parsed_lines) > 1 else ""
            rejected_list = parsed_lines[2] if len(parsed_lines) > 2 else ""
        except Exception as e:
            asked_for_list, chosen_list, rejected_list = "", "", ""

        record = {
            "asked_for_list": asked_for_list,
            "chosen_list": chosen_list,
            "rejected_list": rejected_list
        }
        batch_outputs.append(record)

        # Append to file every 50 iterations
        if (idx + 1) % 50 == 0:
            append_outputs(batch_outputs, _OUTPUT_PATH.value)
            batch_outputs = []

    # Append any remaining records
    if batch_outputs:
        append_outputs(batch_outputs, _OUTPUT_PATH.value)

if __name__ == "__main__":
    app.run(main)
