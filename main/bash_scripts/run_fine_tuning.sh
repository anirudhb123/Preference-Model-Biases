#!/usr/bin/env bash
#SBATCH --job-name=skywork_finetune
#SBATCH --nodelist=nlpgpu[04-10]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0

### ←– only edit this:
SIZE="27B"         # options: 3B, 8B, 27B
BIAS="vagueness"
EXAMPLES=750
EPOCHS=3

# map SIZE → which base model to fine‑tune
case "$SIZE" in
  27B)
    BASE_MODEL_NAME="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    BATCH_SIZE=2
    ;;
  8B)
    BASE_MODEL_NAME="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    BATCH_SIZE=8
    ;;
  7B)
    BASE_MODEL_NAME="ZiyiYe/Con-J-Qwen2-7B"
    BATCH_SIZE=8
    ;;
  3B)
    BASE_MODEL_NAME="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
    BATCH_SIZE=16
    ;;
  2B)
    BASE_MODEL_NAME="Ray2333/GRM-gemma2-2B-rewardmodel-ft"
    BATCH_SIZE=16
    ;;
  *)
    echo "❌ Unknown SIZE '$SIZE'. Valid: 2B, 3B, 7B, 8B, 27B"
    exit 1
    ;;
esac

# lowercase version for naming
LOWER_SIZE="${SIZE,,}"    # e.g. "8b", "3b", "27b"

# derived names
MODEL_REPO_ID="abharadwaj123/skywork-${LOWER_SIZE}-fine-tuned-${BIAS}-${EXAMPLES}-${EPOCHS}"
WANDB_PROJECT="Reward-Model-Biases"
WANDB_RUN_NAME="skywork-${LOWER_SIZE}-fine-tuned-${BIAS}-${EXAMPLES}-${EPOCHS}"

# paths
INPUT_PATH="data/reward_model_counterfactual_data/GEMMA_counterfactuals_${BIAS}.jsonl"
# if you want to use supplemental chatbot arena data, uncomment this line
#SECOND_INPUT_PATH="data/chatbot_arena_labeled_data/chatbot_arena_${BIAS}_labeled.jsonl"

# Activate your scratch venv
source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate
cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh

# caches & tokens
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/
#export HF_TOKEN=""
export WANDB_CACHE_DIR=/nlp/data/wandb_cache/
export WANDB_DATA_DIR=/nlp/data/wandb_data/
#export WANDB_API_KEY=""

# sanity check
echo "Fine‑tuning size=$SIZE on $BASE_MODEL_NAME"
echo "→ batch size will be: $BATCH_SIZE"
echo "→ will push to: $MODEL_REPO_ID"
echo "→ wandb run: $WANDB_RUN_NAME"

# Prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run fine‑tuning
python -u main/counterfactual_fine_tuning.py \
  --input_path="$INPUT_PATH" \
  --second_input_path="$SECOND_INPUT_PATH" \
  --bias="$BIAS" \
  --model_repo_id="$MODEL_REPO_ID" \
  --base_model_name="$BASE_MODEL_NAME" \
  --epochs="$EPOCHS" \
  --batch_size="$BATCH_SIZE" \
  --learning_rate=2e-5 \
  --use_lora=true \
  --lora_r=4 \
  --lora_alpha=8 \
  --validation_split=0.1 \
  --examples="$EXAMPLES" \
  --wandb_project="$WANDB_PROJECT" \
  --wandb_run_name="$WANDB_RUN_NAME"

echo "Fine‑tuning completed. Model pushed to Hugging Face Hub: ${MODEL_REPO_ID}"
