import os
import ast
import time
import json
import argparse
import random
import logging
from tqdm import tqdm
import google.generativeai as genai

import anthropic
import openai
import tenacity

# suppress verbose HTTP logs from OpenAI
logging.getLogger("openai").setLevel(logging.WARNING)

os.environ["GOOGLE_API_KEY"] = # your google api key
os.environ["OPENAI_API_KEY"] = # your openai api key
os.environ["ANTHROPIC_API_KEY"] = # your anthropic api key

# === Prompt template ===
PROMPT_TMPL = """  
You will be given a query issued by a real user to a language model. You will also be given two model responses to this query, and you will need to judge which response is better.

IMPORTANT: You should produce the final judgement as a dictionary in precisely this format (with **): "**output: {{\"judgement\": \"_\"}}**", where you should fill in the spaces with either "Response 1" if Response 1 is better, "Response 2" if Response 2 is better or "Tie" if both responses are equally good or equally bad. Only the three choices "Response 1", "Response 2" and "Tie" are valid. Make note of the ** required to enclose the output dictionary. After generating the output, provide a brief justification of your judgement.

Query: {query}

Response 1: {r1}

Response 2: {r2}

Judgement:
"""

# ─── Model wrappers ─────────────────────────────────────────────

class GPT4:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = "gpt-4o" if model_name.startswith("gpt-4") else model_name
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.organization = os.getenv("OPENAI_ORGANIZATION")

    @tenacity.retry(wait=tenacity.wait_random_exponential(1, 60),
                    stop=tenacity.stop_after_attempt(5))
    def generate(self, prompt, max_len=512):
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_len,
        )
        return resp.choices[0].message.content

class Claude:
    def __init__(self, model_name="claude-3.7-sonnet"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_map = {
            "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        }
        self.model_id = self.model_map.get(model_name, model_name)

    def generate(self, prompt, max_len=512):
        while True:
            try:
                msg = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=max_len,  # Changed from max_tokens_to_sample to max_tokens
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception as e:
                print("Claude error:", e)
                time.sleep(1)


class Gemini:
    def __init__(self, model_name="gemini-2.5-pro"):
        # Initialize the GenAI client
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # Map your alias to the actual preview model name
        if model_name == "gemini-2.5-pro":
            model_name = "gemini-2.5-pro-preview-05-06"
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt, max_len=4096):
        backoff = 1
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
            try:
                generation_config = {
                    "max_output_tokens": max_len,
                } 
                response = self.model.generate_content(
                    contents=prompt,
                    generation_config=generation_config,
                    safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    ]
                )

                # Try the simple .text field
                if getattr(response, "text", None):
                    return response.text

                # Fallback to candidates → parts
                for cand in getattr(response, "candidates", []):
                    for part in getattr(cand.content, "parts", []):
                        if getattr(part, "text", None):
                            return part.text

                print("Warning: response was empty; retrying…")
            
            except Exception as e:
                msg = str(e).lower()
                if "dangerous" in msg or "filter" in msg:
                    print("Gemini refused to generate—flagged as dangerous. Retrying with backoff…")
                else:
                    print(f"Unexpected Gemini error: {e!r}")

            attempt += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

        return "**output: {\"judgement\": \"Tie\"}** Unable to evaluate due to technical issues."

# ─── Extraction utility ─────────────────────────────────────────

def extract_dictionary(text):
    """Find the first {...} and parse it, respecting the required ** markers."""
    try:
        start = text.find('output:')
        if start == -1:
            raise ValueError("Could not find 'output:' prefix")
        json_start = text.find('{', start)
        snippet = text[json_start:]
        brace = 0
        for i, ch in enumerate(snippet):
            if ch == '{': brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    snippet = snippet[:i+1]
                    break
        return ast.literal_eval(snippet)
    except Exception as e:
        print("Parse error:", e, "\nText was:", text)
        return None

# ─── Main batch‐scoring logic ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch‐score pairs with multiple LLMs")
    parser.add_argument(
        "--bias",
        required=True,
        help="Bias name (e.g. length, structure, jargon)"
    )
    args = parser.parse_args()

    # derive input path from bias
    input_path = os.path.join("data", f"{args.bias}_grouped_human_annotations.json")
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    random.seed(0)

    # define models
    models = {
        "gpt-4o": GPT4("gpt-4o"),
        "claude-3.7": Claude("claude-3.7-sonnet"),
        "gemini-2.5": Gemini("gemini-2.5-pro"),
        #"claude-3.5": Claude("claude-3.5-sonnet"),
    }

    # prepare output directory and file handles
    base_dir = os.path.join("data", "generative_model_scores", args.bias)
    os.makedirs(base_dir, exist_ok=True)

    out_files = {}
    for model_name in models:
        safe_name = model_name.replace(".", "-")
        out_path = os.path.join(
            base_dir,
            f"{args.bias}_scored_{safe_name}.jsonl"
        )
        out_files[model_name] = open(out_path, "w", encoding="utf-8")

    # scoring loop
    for ex in tqdm(examples, desc="Scoring examples"):
        prompt = PROMPT_TMPL.format(
            query=ex["query"],
            r1=ex["response_1"],
            r2=ex["response_2"]
        )

        for model_name, mdl in models.items():
            raw = mdl.generate(prompt)
            parsed = extract_dictionary(raw)
            result = {
                "query": ex["query"],
                "response_1": ex["response_1"],
                "response_2": ex["response_2"],
                "raw": raw,
                "judgement": parsed
            }
            out_files[model_name].write(json.dumps(result) + "\n")

    # close files
    for fh in out_files.values():
        fh.close()

    print(f"Wrote {len(examples)} records per model to {base_dir!r}")
    
if __name__ == "__main__":
    main()