#!/bin/bash

cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh
#export OPENAI_API_KEY=""
INPUT_PATH=data/vagueness_baseline.jsonl
BIAS=vagueness
OUTPUT_PATH=data/perturbations/all_data_perturbed_vagueness.jsonl
python3 main/generate_perturbed_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--bias=${BIAS}