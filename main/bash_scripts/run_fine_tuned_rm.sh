#!/usr/bin/env bash
#SBATCH --mem=100G
#SBATCH --nodelist=nlpgpu[04-10]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

SIZE="27B"            # 2B | 3B | 8B | 27B
BIAS="vagueness"
EXAMPLES="750"
USE_ADAPTER="true"     # set to "false" for base model

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

source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/

cd /mnt/nlpgridio3/data/anirudh2/
source main/bash_scripts/set_keys.sh

INPUT_PATH="data/perturbations/all_data_perturbed_${BIAS}.jsonl"

# choose output path based on adapter use
if [ "$USE_ADAPTER" = "true" ]; then
  OUTPUT_PATH="data/fine_tuned_model_scores/${BIAS}/${BIAS}_scored_fine_tuned_${SIZE}_${EXAMPLES}_3.jsonl"
else
  OUTPUT_PATH="data/fine_tuned_model_scores/${BIAS}/${BIAS}_scored_${SIZE}_new_prompt.jsonl"
fi

# build the command array
CMD=(python3 -u main/run_fine_tuned_preference_model.py
     --model_name "$BASE_MODEL_NAME"
     --input_path "$INPUT_PATH"
     --output_path "$OUTPUT_PATH")

# append adapter flag if requested
if [ "$USE_ADAPTER" = "true" ]; then
  size_lc="${SIZE,,}"
  ADAPTER_NAME="abharadwaj123/skywork-${size_lc}-fine-tuned-${BIAS}-${EXAMPLES}-3"
  CMD+=(--adapter_name "$ADAPTER_NAME")
fi

# execute
"${CMD[@]}"
