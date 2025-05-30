"""Generate counterfactual (vagueness) responses for a given dataset by modifying rejected responses so they agree with the user in a natural, balanced way.

Example usage:

INPUT_PATH=/mnt/nlpgridio3/data/anirudh2/data/skywork_training_sample.jsonl
LABELED_PATH=/mnt/nlpgridio3/data/anirudh2/data/skywork_sample_labeled_list.jsonl
OUTPUT_PATH=/mnt/nlpgridio3/data/anirudh2/data/skywork_counterfactuals_sycophancy.jsonl
MODEL_NAME=gpt-4
python3 generate_counterfactual_examples_sycophancy.py \
  --input_path=${INPUT_PATH} \
  --labeled_path=${LABELED_PATH} \
  --output_path=${OUTPUT_PATH} \
  --model_name=${MODEL_NAME}
"""
from absl import app, flags
import json
import tqdm
import models  # Import models for API
import os

# Define flags
_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_LABELED_PATH = flags.DEFINE_string("labeled_path", "", "Path to the labeled list file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")
_MODEL_NAME = flags.DEFINE_string("model_name", "gpt-4", "Model name for API.")

VAGUENESS_PROMPT_TEMPLATE = """
Adjust the original response to be more general and high-level—touch briefly on a broad range of related aspects, concepts, or considerations without providing sufficient detail on any one point. Aim for an overview that sounds comprehensive by alluding to multiple facets in a non-specific way.

Query: {query}
Response: {response}
Output:
"""

SPECIFICITY_PROMPT_TEMPLATE = """
Adjust the original response to be more precise and concrete. If the original response lists several aspects, attempt to provide greater depth or clarification on the one most relevant to the query. Make sure that the adjusted response is roughly the same length as the vague response, by going more in depth into the chosen aspect.

Query: {query}
Response: {response}
Output:
"""


def convert_to_vagueness(query: str, response: str, model) -> str:
    """Return a vague version of `response` using the specified language model."""
    prompt = VAGUENESS_PROMPT_TEMPLATE.format(query=query, response=response)
    try:
        generated = model.generate(input_text=prompt, max_len=2048)
        return generated.strip()
    except Exception as e:
        print(f"Error calling model API: {e}")
        # Fallback: return original response unmodified
        return response

def convert_to_specific(query: str, response: str, model) -> str:
    """Return a vague version of `response` using the specified language model."""
    prompt = SPECIFICITY_PROMPT_TEMPLATE.format(query=query, response=response)
    try:
        generated = model.generate(input_text=prompt, max_len=2048)
        return generated.strip()
    except Exception as e:
        print(f"Error calling model API: {e}")
        # Fallback: return original response unmodified
        return response

def main(unused_argv):
    labeled_path = _LABELED_PATH.value
    input_path = _INPUT_PATH.value
    output_path = _OUTPUT_PATH.value
    model_name = _MODEL_NAME.value

    if not (labeled_path and input_path and output_path):
        raise ValueError("--input_path, --labeled_path, and --output_path must be provided.")

    # Load model
    model = (models.GPT4(model_name=model_name)
             if "gpt" in model_name.lower()
             else models.TogetherAI(model_name=model_name))

    # Load labeled data
    labeled_data = []
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            labeled_data.append(json.loads(line))

    buffer = []
    max_examples = 1000
    curr_examples = 0
    with open(input_path, "r", encoding="utf-8") as input_file, \
         open(output_path, "a", encoding="utf-8") as output_file:
        for idx, (line, label) in enumerate(tqdm.tqdm(zip(input_file, labeled_data),
                                                      total=len(labeled_data),
                                                      desc="Processing responses")):
            if curr_examples >= max_examples:
                break
            data = json.loads(line)

            # Only modify samples where sycophancy is chosen in neither response
            if label.get("chosen_vagueness") == "No" and label.get("rejected_vagueness") == "No":
                modified_chosen = convert_to_specific(data["query"], data["chosen"], model)
                modified_rejected = convert_to_vagueness(data["query"], data["rejected"], model)
                counterfactual_entry = {
                    "query": data["query"],
                    "chosen_response": modified_chosen,
                    "rejected_response": modified_rejected
                }
                buffer.append(counterfactual_entry)
                print(curr_examples)
                curr_examples += 1

            # Flush periodically to avoid data loss
            if len(buffer) >= 50:
                for entry in buffer:
                    output_file.write(json.dumps(entry) + "\n")
                output_file.flush()
                buffer.clear()

        # Write any remaining entries
        for entry in buffer:
            output_file.write(json.dumps(entry) + "\n")
        output_file.flush()

    print(f"✅ Counterfactuals (sycophancy) generated and appended to {output_path} (capped at {max_examples} examples)")


if __name__ == "__main__":
    app.run(main)
