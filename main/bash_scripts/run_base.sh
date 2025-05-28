#!/bin/bash

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/cmalaviya/huggingface_data/

cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh
INPUT_PATH=data/vagueness_queries.jsonl
OUTPUT_PATH=data/vagueness_baseline.jsonl
python3 -u main/generate_base_responses.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH}
