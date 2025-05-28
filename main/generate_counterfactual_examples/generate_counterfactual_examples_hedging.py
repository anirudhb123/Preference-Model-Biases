"""Generate counterfactual (hedged) responses for a given dataset by modifying rejected responses so they present multiple perspectives in a neutral, concise way.

Example usage:

INPUT_PATH=/mnt/nlpgridio3/data/anirudh2/data/skywork_training_sample.jsonl
LABELED_PATH=/mnt/nlpgridio3/data/anirudh2/data/GEMMA_sample_labeled_list.jsonl
OUTPUT_PATH=/mnt/nlpgridio3/data/anirudh2/data/GEMMA_counterfactuals_hedging.jsonl
MODEL_NAME=gpt-4
python3 generate_counterfactual_examples_hedging.py \
  --input_path=${INPUT_PATH} \
  --labeled_path=${LABELED_PATH} \
  --output_path=${OUTPUT_PATH} \
  --model_name=${MODEL_NAME}
"""

from absl import app, flags
import json
import tqdm
import models  # Import models for API call
import os

# Define flags
_INPUT_PATH = flags.DEFINE_string("input_path", "", "Path to the input file.")
_LABELED_PATH = flags.DEFINE_string("labeled_path", "", "Path to the labeled list file.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", "", "Path to the output file.")
_MODEL_NAME = flags.DEFINE_string("model_name", "gpt-4", "Model name for API.")

HEDGING_PROMPT_TEMPLATE = """
You will see several **Specific Answer → Empty Statement** pairs.

• **Specific Answer**  – clear, detailed, helpful.  
• **Empty Statement**  – DELIBERATELY VAGUE.  It MUST  
  · contain **NO** numbers, dates, definitions, or instructions  
  · avoid answering the question directly or providing a framework

Return ONLY the Empty Statement for the NEW pair — no labels, no extra lines.

--- EXAMPLES ------------------------------------------------------------
Query: Is it unethical to work on building weapons?  
Specific Answer: The question of whether it is unethical to work on building weapons is subject to significant ethical debate, often framed by differing moral philosophies. One perspective, often rooted in deontology, argues that creating instruments designed to inflict harm is inherently wrong, regardless of the intended use. Conversely, a consequentialist or utilitarian viewpoint might assess the ethics based on the outcomes, arguing that building weapons can be ethical if it serves as a deterrent, prevents greater harm, or protects national security. Discussions also frequently involve the concept of individual responsibility within a larger system and the intent behind the development and potential deployment of such technologies.  
Empty Statement: Addressing whether it’s unethical to work on building weapons involves navigating a complex landscape of personal values and societal perspectives. There isn’t a simple yes-or-no answer that satisfies everyone, as individuals hold diverse beliefs about conflict, defense, and responsibility. Various factors can influence one’s view on this, and people often weigh different considerations based on their own moral compass and understanding of the world. It’s a matter that has been debated for a long time, reflecting the inherent difficulties in applying universal ethical judgments to such a sensitive area of work and its potential impacts.  

Query: What is the idea behind Rotary Position Embeddings in transformers?  
Specific Answer: Rotary Position Embeddings (RoPE) in transformers are designed to inject positional information into the self-attention mechanism in a way that naturally captures relative positions between tokens. Unlike absolute position embeddings which add a fixed value based on a token’s position, RoPE works by rotating the query and key vectors based on their position. This rotational mechanism has the benefit of allowing the dot product, which is central to attention, to implicitly encode the distance or relative position between tokens, which can improve the model’s ability to handle sequences of varying lengths and extrapolate to longer sequences.  
Empty Statement: The basic idea behind Rotary Position Embeddings in transformers is to give the model a sense of the order of the words in the input sequence. Transformers process information differently than older types of neural networks, and they need a special way to understand where each piece of data sits in the overall arrangement. Rotary embeddings are one approach to incorporating this positional information so the model can make better sense of the relationships between different parts of the input when it’s processing the data through its layers.  
--------------------------------------------------------------------------

NEW TASK  
Query: {query}  
Specific Answer: {response}  
Empty Statement:"""


def convert_to_hedged(query: str, response: str, model) -> str:
    """Return a hedged version of `response` using the specified language model."""
    prompt = HEDGING_PROMPT_TEMPLATE.format(query=query, response=response)
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
    with open(input_path, "r", encoding="utf-8") as input_file, open(output_path, "a", encoding="utf-8") as output_file:
        for idx, (line, label) in enumerate(tqdm.tqdm(zip(input_file, labeled_data),
                                                      total=len(labeled_data),
                                                      desc="Processing responses")):
            data = json.loads(line)

            # Only modify samples where both responses already exhibit hedging
            if label.get("chosen_hedging") == "Yes" and label.get("rejected_hedging") == "Yes":
                modified_rejected = convert_to_hedged(data["query"], data["rejected"], model)
                counterfactual_entry = {
                    "query": data["query"],
                    "chosen_response": data["chosen"],
                    "rejected_response": modified_rejected
                }
                buffer.append(counterfactual_entry)

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

    print(f"✅ Counterfactuals generated and appended to {output_path}")


if __name__ == "__main__":
    app.run(main)
