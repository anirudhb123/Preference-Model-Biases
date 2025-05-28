#!/bin/bash
#SBATCH --mem=100G        
#SBATCH --nodelist=nlpgpu[04-09]   
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1    

SIZE="27B"           # options: 2B, 3B, 7B, 8B, 27B
BIAS="vagueness,jargon,length"
EXAMPLES="500"
USE_ADAPTER="true" 

# normalize bias list into underscored form
COMBINED_BIASES="${BIAS//,/_}"   # e.g. vagueness_length_jargon

# pick base model & batch size
case "$SIZE" in
  27B) BASE_MODEL_NAME="Skywork/Skywork-Reward-Gemma-2-27B-v0.2"; BATCH_SIZE=8 ;;
  8B)  BASE_MODEL_NAME="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"; BATCH_SIZE=8 ;;
  7B)  BASE_MODEL_NAME="ZiyiYe/Con-J-Qwen2-7B";              BATCH_SIZE=8 ;;
  3B)  BASE_MODEL_NAME="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"; BATCH_SIZE=16 ;;
  2B)  BASE_MODEL_NAME="Ray2333/GRM-gemma2-2B-rewardmodel-ft";  BATCH_SIZE=16 ;;
  *)
    echo "❌ Unknown SIZE '$SIZE'. Valid: 2B, 3B, 7B, 8B, 27B"
    exit 1
    ;;
esac

# environment setup
source /mnt/nlpgridio3/data/anirudh2/venv/bin/activate
export TRITON_CACHE_DIR=/mnt/nlpgridio3/data/anirudh2/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
export TRANSFORMERS_CACHE=/nlp/data/huggingface_cache/
export HF_HOME=/nlp/data/huggingface_cache/
export HF_DATASETS_CACHE=/mnt/nlpgridio3/data/anirudh2/huggingface_data/
cd /mnt/nlpgridio3/data/anirudh2/ || exit

# set up output directory & file
if [ "$USE_ADAPTER" = "true" ]; then
  OUTPUT_DIR="data/rewardbench_results/${COMBINED_BIASES}"
else
  OUTPUT_DIR="data/rewardbench_results/base-model"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/${SIZE}_${COMBINED_BIASES}_${EXAMPLES}_rewardbench.out"

# build command
CMD=(python -u main/rewardbench_eval.py
     --model "$BASE_MODEL_NAME"
     --batch_size "$BATCH_SIZE"
     --max_length 512
     --torch_dtype bfloat16)

if [ "$USE_ADAPTER" = "true" ]; then
  size_lc="${SIZE,,}"
  ADAPTER_NAME="abharadwaj123/skywork-${size_lc}-fine-tuned-${COMBINED_BIASES}-${EXAMPLES}-3"
  CMD+=(--peft_adapter "$ADAPTER_NAME")
fi

# run evaluation
echo "Running rewardbench eval → $OUTPUT_FILE"
"${CMD[@]}" &> "$OUTPUT_FILE"
echo "Done. Results saved to $OUTPUT_FILE"
