r"""Generate a base response for a given set of queries.

Example usage:

INPUT_PATH=data/all_data.jsonl
OUTPUT_PATH=data/all_data_baseline.jsonl
python3 main/generate_base_responses.py \
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
from data_utils import tsv_utils
import tqdm


_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4o", "Model name."
)


def main(unused_argv) -> None:
    base_prompt = "\n".join(tsv_utils.read_txt(f"main/prompts/original_response_prompt.txt"))
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
    idx = 0
    for ex in tqdm.tqdm(examples[:500]):
        cur_prompt = (base_prompt + ".")[:-1]
        cur_prompt = cur_prompt.replace("[QUERY]", ex["query"])
        generated_response = model.generate(input_text=cur_prompt, max_len=2048)
        ex["original_response"] = generated_response
        outputs.append(ex)
        idx += 1

    jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
  app.run(main)
