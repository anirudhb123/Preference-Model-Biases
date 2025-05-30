#!/bin/bash
#SBATCH --gres=gpu:1              
#SBATCH --nodelist=nlpgpu[01-10]
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

MODEL_NAME="gpt-4"
BIAS="vagueness"
INPUT_PATH="data/skywork_training_sample.jsonl"
LABELED_PATH="data/reward_model_training_labeled_data/skywork_sample_labeled_${BIAS}.jsonl"
OUTPUT_PATH="data/reward_model_counterfactual_data/skywork_counterfactuals_${BIAS}.jsonl"

# swap out approprpriate bias
python3 -u main/generate_counterfactual_examples/generate_counterfactual_examples_${BIAS}.py \
  --input_path=${INPUT_PATH} \
  --labeled_path=${LABELED_PATH} \
  --output_path=${OUTPUT_PATH} \
  --model_name=${MODEL_NAME}
