#!/usr/bin/env python3
r"""Generate counterfactual responses for a given dataset by extending rejected answers,
but cap at 1000 generated examples."""

from absl import app, flags
import json, tqdm, models

_INPUT_PATH   = flags.DEFINE_string("input_path",  "", "Path to the input file.")
_OUTPUT_PATH  = flags.DEFINE_string("output_path", "", "Path to the output file.")
_LABELED_PATH = flags.DEFINE_string("labeled_path", "", "(ignored) passed by the bash script")
_MODEL_NAME   = flags.DEFINE_string("model_name", "gpt-4", "Model name for API.")
_THRESHOLD    = flags.DEFINE_float("threshold",    1.5,     "Min (extended_len/chosen_len) ratio.")
_CAP          = flags.DEFINE_integer("cap",       1000,    "Max number of counterfactuals to generate.")

flags.mark_flag_as_required("input_path")
flags.mark_flag_as_required("output_path")
flags.mark_flag_as_required("labeled_path")

def extend_short_answer(query: str, short_ans: str, model) -> str:
    prompt = f"""Adjust the original response to make it longer, but change *nothing* else.
Query: {query}
Response: {short_ans}
Output:"""
    try:
        return model.generate(input_text=prompt, max_len=2048).strip()
    except Exception as e:
        print("🔴 API error:", e)
        return short_ans


def main(argv):
    # 1) load the LLM
    mname = _MODEL_NAME.value.lower()
    model = (models.GPT4(model_name=mname)
             if "gpt" in mname else
             models.TogetherAI(model_name=mname))

    inp    = _INPUT_PATH.value
    out    = _OUTPUT_PATH.value
    thresh = _THRESHOLD.value
    cap    = _CAP.value

    written = 0
    with open(inp,  'r', encoding='utf-8') as fin, \
         open(out, 'w', encoding='utf-8', buffering=1)  as fout:

        pbar = tqdm.tqdm(fin, desc="Generating CF examples")
        for line in pbar:
            rec = json.loads(line)
            q        = rec["query"]
            chosen   = rec["chosen"]
            rejected = rec["rejected"]

            # only process if chosen is longer than rejected
            len_chosen = len(chosen.split())
            len_rej    = len(rejected.split())
            if len_chosen <= len_rej:
                continue

            # extend the rejected (short) answer
            extended = extend_short_answer(q, rejected, model)

            # measure ratio vs chosen
            len_ext = len(extended.split())
            ratio   = len_ext / len_chosen if len_chosen else 0.0

            print(f"RATIO: {ratio:.2f}")
            if ratio < thresh:
                continue

            # emit counterfactual
            fout.write(json.dumps({
                "query":             q,
                "chosen_response":   chosen,
                "rejected_response": extended,
                "ratio":             round(ratio, 2)
            }) + "\n")

            written += 1

            # update the bar with both written count and last ratio
            pbar.set_postfix(written=written, ratio=f"{ratio:.2f}")

            # print the ratio immediately
            tqdm.tqdm.write(f"🔹 Wrote example #{written}, ratio={ratio:.2f}")

            # every 50 writes, log an interim summary
            if written % 50 == 0:
                tqdm.tqdm.write(f"🔹 {written} examples so far… latest ratio={ratio:.2f}")

            # stop at cap
            if written >= cap:
                tqdm.tqdm.write(f"✅ Reached cap of {cap} examples, stopping.")
                return

        # final summary
        tqdm.tqdm.write(f"✅ Done! Counterfactuals written: {written} examples to {out}")

if __name__ == "__main__":
    app.run(main)