#!/bin/bash
#SBATCH --gres=gpu:1     
#SBATCH --nodes=1              
#SBATCH --time=3-0

cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh
#export OPENAI_API_KEY=""
BIAS=sycophancy
INPUT_PATH=data/skywork_training_sample.jsonl
OUTPUT_PATH=data/chatbot_arena_labeled_data/chatbot_arena_${BIAS}_labeled.jsonl
python3 -u main/data_labeling/train/${BIAS}_labeling.py \
--input_path=${INPUT_PATH} \
--output_path=${OUTPUT_PATH} \
--model_name=gpt-4
