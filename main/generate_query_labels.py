r"""Classify queries into predefined categories.

Example usage:

INPUT_PATH=data/all_data.jsonl
OUTPUT_PATH=data/all_data_labeled.jsonl
python3 main/generate_query_labels.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
"""

from absl import app
from absl import flags

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from data_utils import jsonl_utils
import tqdm


_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing queries."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file for classified queries."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)


def main(unused_argv) -> None:
    classification_prompt = """
You are a query classifier. Your task is to classify queries into one of the following categories: 
Closed-ended, Open-ended, Opinion-based, Procedural, Comparative, Hypothetical, Technical, Ambiguous, or Adversarial.
Provide only the category name as the output.

Query: [QUERY]
Category:
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

    outputs = []
    for ex in tqdm.tqdm(examples):
        cur_prompt = classification_prompt.replace("[QUERY]", ex["query"])

        generated_category = model.generate(input_text=cur_prompt, max_len=50).strip()

        ex["category"] = generated_category
        outputs.append(ex)

    jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)
