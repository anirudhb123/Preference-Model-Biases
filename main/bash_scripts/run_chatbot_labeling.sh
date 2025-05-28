#!/bin/bash
#SBATCH --nodelist=nlpgpu[01-10]   
#SBATCH --gres=gpu:1     
#SBATCH --nodes=1              
#SBATCH --time=3-0

# Set up cache directories.
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p $TRITON_CACHE_DIR

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/
export HUGGINGFACE_HUB_TOKEN=""

# Change to your working directory.
cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh

#export OPENAI_API_KEY=""
BIAS="vagueness"
DATASET_NAME="lmsys/chatbot_arena_conversations"
OUTPUT_PATH=data/chatbot_arena_labeled_data/chatbot_arena_${BIAS}_labeled.jsonl
python3 -u main/data_labeling/chatbot_arena/${BIAS}_labeling_chatbot_arena.py \
--dataset_name=${DATASET_NAME} \
--output_path=${OUTPUT_PATH} \
--model_name=gpt-4
