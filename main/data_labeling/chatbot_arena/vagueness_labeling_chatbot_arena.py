import json
import os
import sys
from absl import app, flags
import tqdm
from datasets import load_dataset

# Update the Python path to include your project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models

# Flags for the dataset, output, and model name.
_DATASET_NAME = flags.DEFINE_string(
    "dataset_name", "lmsys/chatbot_arena_conversations", "Hugging Face dataset name"
)
_SPLIT = flags.DEFINE_string("split", "train", "Dataset split to use")
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output file with labeled vagueness-related classifications."
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gpt-4", "Model name to use for classification."
)

def append_outputs(outputs, output_path):
    """Append a list of output records to the file in JSONL format."""
    with open(output_path, "a") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

def extract_query(conversation_a, conversation_b):
    """
    Extract the query from the two conversations.
    It assumes that the query is the first message with role "user" found in either conversation.
    """
    for conv in (conversation_a, conversation_b):
        for msg in conv:
            if msg.get("role", "").lower() == "user":
                return msg.get("content", "")
    # Fallback: if no user role is found, return the first message of conversation_a
    return conversation_a[0].get("content", "") if conversation_a else ""

def extract_response(conversation):
    """
    Concatenate the "content" fields from all messages in the conversation.
    """
    return "\n".join(msg.get("content", "") for msg in conversation)

def main(unused_argv) -> None:
    # Define the classification prompt template.
    classification_prompt = """
You are a query classifier. Your task is to classify the following query and responses into three categories:

1. **Query Classification:** Decide whether the query is Technical or Non-Technical.  
   - **Non-Technical**: general or everyday questions that do not demand such specialized expertise.  
   - **Technical**: deeply technical queries that require specialized expertise or detailed domain knowledge (e.g., algorithmic, engineering, scientific, or programming questions).  
   Provide only one word—either "Non-Technical" or “Technical" the output.

2. **Chosen Response Contains Vagueness:** Does the chosen response predominantly use general, high-level statements that allude to many aspects without explaining any of them fully? (Yes/No)

3. **Rejected Response Contains Vagueness:** Does the rejected response predominantly use general, high-level statements that allude to many aspects without explaining any of them fully? (Yes/No)

Here are the query and responses:

- **Query:** [QUERY]  
- **Chosen Response:** [CHOSEN]  
- **Rejected Response:** [REJECTED]  

Provide your answers in the following format:  
Query Classification: [Classification]  
Chosen contains Vagueness: [Yes/No]  
Rejected contains Vagueness: [Yes/No]
"""

    # Load the Hugging Face dataset.
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    dataset = load_dataset(_DATASET_NAME.value, split=_SPLIT.value, use_auth_token=token)

    # ----------------------------------------------------------------
    # Build remaining_queries from your all_data / baseline files
    full_query_set = set()
    with open("data/full_query_set.jsonl", "r") as f:
        for line in f:
            if line.strip():
                full_query_set.add(json.loads(line)["query"])
    used_query_set = set()
    with open("data/all_data_baseline.jsonl", "r") as f:
        for line in f:
            if line.strip():
                used_query_set.add(json.loads(line)["query"])
    remaining_queries = full_query_set - used_query_set
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Instantiate your model
    model_name = _MODEL_NAME.value.lower()
    if "gpt" in model_name:
        model = models.GPT4(model_name=_MODEL_NAME.value)
    elif "gemini" in model_name:
        model = models.Gemini(model_name=_MODEL_NAME.value)
    elif "claude" in model_name:
        model = models.Claude(model_name=_MODEL_NAME.value)
    elif "jamba" in model_name:
        model = models.Jamba(model_name=_MODEL_NAME.value)
    else:
        model = models.TogetherAI(model_name=_MODEL_NAME.value)
    # ----------------------------------------------------------------

    # Read existing outputs to avoid duplicates
    existing_queries = set()
    if os.path.exists(_OUTPUT_PATH.value):
        with open(_OUTPUT_PATH.value, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing_queries.add(rec["query"])

    # Counters for *new* additions in this run
    new_yes_yes = 0
    new_yes_no  = 0
    new_no_no = 0

    batch_outputs = []
    for idx, ex in enumerate(tqdm.tqdm(dataset)):
        # Stop only when we've hit both new targets
        if new_yes_yes >= 250 and new_yes_no >= 250 and new_no_no >= 250:
            break

        # Extract query and skip if already processed or not in remaining set
        query = extract_query(ex["conversation_a"], ex["conversation_b"])
        if query in existing_queries or query not in remaining_queries:
            continue

        # Determine chosen vs rejected
        if ex.get("winner") == "model_b":
            chosen_conv, rejected_conv = ex["conversation_b"], ex["conversation_a"]
        else:
            chosen_conv, rejected_conv = ex["conversation_a"], ex["conversation_b"]

        chosen_response = extract_response(chosen_conv)
        rejected_response = extract_response(rejected_conv)

        # Build prompt & call the model
        cur_prompt = (classification_prompt
                      .replace("[QUERY]", query)
                      .replace("[CHOSEN]", chosen_response)
                      .replace("[REJECTED]", rejected_response))
        gen = model.generate(input_text=cur_prompt, max_len=100).strip()

        # Parse the three labels
        parts = [line.split(": ")[1].strip() for line in gen.splitlines() if ": " in line]
        asked_for_vagueness  = parts[0] if len(parts) > 0 else ""
        chosen_vagueness     = parts[1] if len(parts) > 1 else ""
        rejected_vagueness   = parts[2] if len(parts) > 2 else ""

        # Only append into our two buckets, up to the new-target
        # Only append into our two buckets, up to the new-target
        chosen_lower = chosen_vagueness.lower()
        rejected_lower = rejected_vagueness.lower()

        print((chosen_lower, rejected_lower))

        if "yes" in chosen_lower and "yes" in rejected_lower:
            if new_yes_yes >= 250:
                continue
            new_yes_yes += 1
            norm_chosen = "Yes"
            norm_rejected = "Yes"
        elif "yes" in chosen_lower and "no" in rejected_lower:
            if new_yes_no >= 250:
                continue
            new_yes_no += 1
            norm_chosen = "Yes"
            norm_rejected = "No"
        elif "no" in chosen_lower and "no" in rejected_lower:
            if new_no_no >= 250:
                continue
            new_no_no += 1
            norm_chosen = "No"
            norm_rejected = "No"
        else:
            continue

        # Build record and queue for write, using the normalized labels
        record = {
            "query": query,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response,
            "query_classification": asked_for_vagueness,
            "chosen_vagueness": norm_chosen,
            "rejected_vagueness": norm_rejected
        }
        batch_outputs.append(record)
        existing_queries.add(query)

        # Flush every 50
        if len(batch_outputs) >= 50:
            append_outputs(batch_outputs, _OUTPUT_PATH.value)
            batch_outputs = []

    # Write any leftovers
    if batch_outputs:
        append_outputs(batch_outputs, _OUTPUT_PATH.value)

    print(f"Done. New Yes/Yes added: {new_yes_yes}, New Yes/No added: {new_yes_no}, New No/No added: {new_no_no}")


if __name__ == "__main__":
    app.run(main)
