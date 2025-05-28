r"""Analyze queries and classify responses into vagueness-related categories.

Example usage:

INPUT_PATH=data/skywork_training_sample.jsonl
OUTPUT_PATH=data/GEMMA_sample_labeled.jsonl
python3 main/vagueness_labeling.py \
    --input_path=${INPUT_PATH} \
    --output_path=${OUTPUT_PATH} \
    --model_name=gpt-4
"""

from absl import app
from absl import flags
import logging
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
    "output_path", "", "Path to the output file with labeled vagueness-related classifications."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)


def main(unused_argv) -> None:
    logging.basicConfig(level=logging.INFO)
    classification_prompt = """
You are a query classifier. Your task is to classify the following query and responses into five categories:

1. **Query Classification:** Decide whether the query is Technical or Non-Technical.  
   - **Non-Technical**: general or everyday questions that do not demand such specialized expertise.  
   - **Technical**: deeply technical queries that require specialized expertise or detailed domain knowledge (e.g., algorithmic, engineering, scientific, or programming questions).  
   Provide only one word—either "Non-Technical" or "Technical".

2. **Chosen Response Contains Specificity:** Does the chosen response use concrete, detailed language focused on the single most relevant aspect rather than broad, high-level statements? (Yes/No)

3. **Chosen Response Contains Vagueness:** Does the chosen response predominantly use general, high-level statements that allude to many aspects without explaining any of them fully? (Yes/No)

4. **Rejected Response Contains Specificity:** Does the rejected response use concrete, detailed language focused on the single most relevant aspect rather than broad, high-level statements? (Yes/No)

5. **Rejected Response Contains Vagueness:** Does the rejected response predominantly use general, high-level statements that allude to many aspects without explaining any of them fully? (Yes/No)

Here are the query and responses:

- **Query:** [QUERY]  
- **Chosen Response:** [CHOSEN]  
- **Rejected Response:** [REJECTED]  

Provide your answers in the following format:
Query Classification: [Classification]
Chosen contains Specificity: [Yes/No]
Chosen contains Vagueness: [Yes/No]
Rejected contains Specificity: [Yes/No]
Rejected contains Vagueness: [Yes/No]
"""

    examples = jsonl_utils.read(_INPUT_PATH.value)

    # Select model based on name
    name = _MODEL_NAME.value.lower()
    if "gpt" in name:
        model = models.GPT4(model_name=_MODEL_NAME.value)
    elif "gemini" in name:
        model = models.Gemini(model_name=_MODEL_NAME.value)
    elif "claude" in name:
        model = models.Claude(model_name=_MODEL_NAME.value)
    elif "jamba" in name:
        model = models.Jamba(model_name=_MODEL_NAME.value)
    else:
        model = models.TogetherAI(model_name=_MODEL_NAME.value)

    outputs = []
    for idx, ex in enumerate(tqdm.tqdm(examples[:2500])):
        cur_prompt = (
            classification_prompt
            .replace("[QUERY]", ex.get("query", ""))
            .replace("[CHOSEN]", ex.get("chosen", ""))
            .replace("[REJECTED]", ex.get("rejected", ""))
        )

        generated_output = model.generate(input_text=cur_prompt, max_len=100).strip()

        # Initialize defaults
        query_classification = ""
        chosen_specificity = ""
        chosen_vagueness = ""
        rejected_specificity = ""
        rejected_vagueness = ""

        # Robust parsing by label matching
        for line in generated_output.splitlines():
            line = line.strip()
            if line.lower().startswith("query classification:"):
                query_classification = line.split(":", 1)[1].strip()
            elif line.lower().startswith("chosen contains specificity:"):
                chosen_specificity = line.split(":", 1)[1].strip()
            elif line.lower().startswith("chosen contains vagueness:"):
                chosen_vagueness = line.split(":", 1)[1].strip()
            elif line.lower().startswith("rejected contains specificity:"):
                rejected_specificity = line.split(":", 1)[1].strip()
            elif line.lower().startswith("rejected contains vagueness:"):
                rejected_vagueness = line.split(":", 1)[1].strip()

        # Warn if any field is missing
        if not all([query_classification, chosen_specificity, chosen_vagueness, rejected_specificity, rejected_vagueness]):
            logging.warning(f"Incomplete parse for example {idx}: {generated_output!r}")

        outputs.append({
            "query_classification": query_classification,
            "chosen_specificity": chosen_specificity,
            "chosen_vagueness": chosen_vagueness,
            "rejected_specificity": rejected_specificity,
            "rejected_vagueness": rejected_vagueness
        })

        # Periodically flush every 50 examples
        if (idx + 1) % 50 == 0:
            jsonl_utils.write(_OUTPUT_PATH.value, outputs)

    # Write any remaining outputs
    if outputs and (len(outputs) % 50) != 0:
        jsonl_utils.write(_OUTPUT_PATH.value, outputs)


if __name__ == "__main__":
    app.run(main)
