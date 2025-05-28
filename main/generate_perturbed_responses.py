r"""Generate a perturbed response for a given set of queries.

Example usage:

INPUT_PATH=data/all_data_baseline_test.jsonl
BIAS=elaboration
OUTPUT_PATH=data/all_data_perturbed_${BIAS}_test.jsonl
python3 main/generate_perturbed_responses.py \
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
_BIAS = flags.DEFINE_string(
    "bias", "", "Bias that perturubation applies."
)


def main(unused_argv) -> None:
    rewrite_prompt = "\n".join(tsv_utils.read_txt(f"main/prompts/{_BIAS.value}_rewrite_prompt.txt"))
    rerewrite_prompt = "\n".join(tsv_utils.read_txt(f"main/prompts/{_BIAS.value}_re_rewrite_prompt.txt"))
    examples = jsonl_utils.read(_INPUT_PATH.value)

    # sycophancy uses separate queries
    if _BIAS.value == "sycophancy": # or _BIAS.value == "hedging":
        examples = jsonl_utils.read("data/sycophancy_data_baseline.jsonl")

    #elif _BIAS.value == "hedging":
    #    labels = jsonl_utils.read("data/all_data_labeled.jsonl")
    #    labels = [rec["category"] for rec in labels]

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
    batch_size = 50 

    for i, ex in enumerate(tqdm.tqdm(examples)):
        if _BIAS.value == "hedging":
            #if labels[i] != "Open-ended" and labels[i] != "Opinion-based" and labels[i] != "Technical":
            #    continue
            pass
        cur_prompt = rewrite_prompt.replace("[QUERY]", ex["query"]).replace("[RESPONSE]", ex["original_response"])
        generated_response = model.generate(input_text=cur_prompt, max_len=2048)
        ex["perturbed_response"] = generated_response.strip()  

        cur_prompt = rerewrite_prompt.replace("[QUERY]", ex["query"]).replace("[RESPONSE]", ex["perturbed_response"])
        generated_response = model.generate(input_text=cur_prompt, max_len=2048)
        ex["base_response"] = generated_response.strip()
        outputs.append(ex)

        idx += 1
        if idx % batch_size == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)
    
    if idx % batch_size != 0:
        jsonl_utils.write(_OUTPUT_PATH.value, outputs)

if __name__ == "__main__":
  app.run(main)
