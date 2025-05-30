"""Generate counterfactual (sycophantic) responses for a given dataset by modifying rejected responses so they agree with the user in a natural, balanced way.

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

SYCOPHANCY_PROMPT_TEMPLATE = """
Adjust the original response to be more agreeable with the user. This can involve validating their perspective and mirroring their language. Make sure to not be over the top.

Query: {query}
Response: {response}
Output:"""


def convert_to_sycophantic(query: str, response: str, model) -> str:
    """Return a sycophantic version of `response` using the specified language model."""
    prompt = SYCOPHANCY_PROMPT_TEMPLATE.format(query=query, response=response)
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
            if label.get("chosen_sycophantic") == "No" and label.get("rejected_sycophantic") == "No":
                modified_rejected = convert_to_sycophantic(data["query"], data["rejected"], model)
                counterfactual_entry = {
                    "query": data["query"],
                    "chosen_response": data.get("chosen"),
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
