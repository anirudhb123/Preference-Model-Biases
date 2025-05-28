from absl import app
from absl import flags

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from data_utils import jsonl_utils
import tqdm

_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing queries and responses."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file with labeled jargon-related classifications."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)

def main(unused_argv) -> None:
    classification_prompt = """
You are a query classifier. Your task is to classify the following query and responses into three categories:

1. **Query Classification:** Decide whether the query is Technical or Non‑Technical.
Technical queries require expertise in a specific domain.
Provide only one word—either “Technical” or “Non‑Technical”—as the output.

2. **Chosen Response Contains Jargon:** Does the chosen response predominantly consist of jargon? (Yes/No)
   *For this classification, consider a response to predominantly contain jargon only if specialized terms or technical language are used so extensively that they dominate the response and make it largely inaccessible to a non-specialist.*

3. **Rejected Response Contains Jargon:** Does the rejected response predominantly consist of jargon? (Yes/No)
   *Apply the same qualitative standard: only classify as Yes if the response is saturated with specialized language to the point that a general audience would struggle to understand the majority of the content.*

Here are the query and responses:

- **Query:** [QUERY]
- **Chosen Response:** [CHOSEN]
- **Rejected Response:** [REJECTED]

Provide your answers in the following format:
Query Classification: [Classification]
Chosen contains Jargon: [Yes/No]
Rejected contains Jargon: [Yes/No]
"""

    # If the output file already exists, load its content, otherwise initialize an empty list.
    outputs = []
    if os.path.exists(_OUTPUT_PATH.value):
        try:
            outputs = jsonl_utils.read(_OUTPUT_PATH.value)
        except Exception as e:
            print("Warning: Could not read the existing output file. Starting with an empty list.")

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

    # Process new examples and append results.
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
            query_classification = parsed_lines[0] if len(parsed_lines) > 0 else ""
            chosen_jargon = parsed_lines[1] if len(parsed_lines) > 1 else ""
            rejected_jargon = parsed_lines[2] if len(parsed_lines) > 2 else ""
        except Exception as e:
            query_classification, chosen_jargon, rejected_jargon = "", "", ""

        outputs.append({
            "query_classification": query_classification,
            "chosen_jargon": chosen_jargon,
            "rejected_jargon": rejected_jargon
        })

        # Write the appended outputs every 50 examples to avoid data loss on long runs.
        if (idx + 1) % 50 == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)

    # Write any remaining outputs after the loop.
    jsonl_utils.write(_OUTPUT_PATH.value, outputs)

if __name__ == "__main__":
    app.run(main)
