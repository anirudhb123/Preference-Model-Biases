import argparse
import logging
import os
import sys
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.utils.data import DataLoader
from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from functools import partial
import torch.multiprocessing as mp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login
    _login(token=HF_TOKEN, add_to_git_credential=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--peft_adapter", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--chat_template", type=str, default="tulu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--do_not_save", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--pref_sets", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--disable_beaker_save", action="store_true")
    parser.add_argument("--quantized", default=True)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args

def process_batch(batch, reward_pipe, logger=None):
    """Process a batch of examples and return scores"""
    chosen_texts = batch["text_chosen"]
    rejected_texts = batch["text_rejected"]
    
    try:
        # Process chosen and rejected texts in parallel batches
        outputs_chosen = reward_pipe(chosen_texts)
        outputs_rejected = reward_pipe(rejected_texts)
        
        # Extract scores
        scores_chosen = [output["score"] for output in outputs_chosen]
        scores_rejected = [output["score"] for output in outputs_rejected]
        
        # Determine results (1 if chosen > rejected, 0 otherwise)
        results = [1 if c > r else 0 for c, r in zip(scores_chosen, scores_rejected)]
        
        return scores_chosen, scores_rejected, results
    except Exception as e:
        if logger:
            logger.error(f"Error processing batch: {str(e)}")
        # Return default values in case of error
        return [0] * len(chosen_texts), [0] * len(rejected_texts), [0] * len(chosen_texts)

class RewardDataset(torch.utils.data.Dataset):
    """Dataset wrapper for parallel processing"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    return {
        "text_chosen": [item["text_chosen"] for item in batch],
        "text_rejected": [item["text_rejected"] for item in batch],
    }

def main():
    args = get_args()
    # Initialize distributed environment
    accelerator = Accelerator()
    logger = get_logger(__name__)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Loading model: {args.model}")
    attn_implementation = args.attn_implementation or "eager"
    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": attn_implementation,
    }

    if args.peft_adapter:
        logger.info(f"Loading PEFT adapter: {args.peft_adapter}")
        peft_config = PeftConfig.from_pretrained(args.peft_adapter)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            torch_dtype=args.torch_dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            num_labels=1,
            # load_in_8bit=True
        )
        # Swap in the specified adapter weights
        model = PeftModel.from_pretrained(
            base_model,
            args.peft_adapter
        )
    # base model
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            torch_dtype=args.torch_dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            num_labels=1,
            # load_in_8bit=True
        )

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=get_conv_template(args.chat_template),
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )
    dataset = dataset.remove_columns("id")
    
    # For a subset evaluation, choose a consistent subset size
    subset_size = len(dataset)
    subset_dataset = dataset.select(range(subset_size))
    subset_subsets = subsets[:subset_size]

    # Remove cache (needed for Gemma 2B)
    if isinstance(model, PeftModel):
        inner = model.base_model
    else:
        inner = model

    # disable the broken hybrid cache
    inner.config.cache_implementation = None
    inner.config.use_cache = False

    # Create reward pipeline
    reward_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        torch_dtype=model_kwargs["torch_dtype"],
        max_length=args.max_length,
        truncation=True,
        padding=True,
        function_to_apply="none"
    )

    # Prepare dataset for parallel processing
    eval_dataset = RewardDataset(subset_dataset)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Use accelerator to prepare the data loader
    dataloader = accelerator.prepare(dataloader)
    
    # Process batches in parallel
    all_scores_chosen = []
    all_scores_rejected = []
    all_results = []
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        scores_chosen, scores_rejected, results = process_batch(batch, reward_pipe, logger)
        all_scores_chosen.extend(scores_chosen)
        all_scores_rejected.extend(scores_rejected)
        all_results.extend(results)

    # Add columns to the subset_dataset (which has the same number of rows as our processed results)
    out_dataset = subset_dataset.add_column("results", all_results[:len(subset_dataset)])
    out_dataset = out_dataset.add_column("subset", subset_subsets)
    out_dataset = out_dataset.add_column("scores_chosen", all_scores_chosen[:len(subset_dataset)])
    out_dataset = out_dataset.add_column("scores_rejected", all_scores_rejected[:len(subset_dataset)])

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Only the main process should handle results aggregation and saving
    if accelerator.is_main_process:
        results_grouped = {"model": args.model, "model_type": "PEFT-LoRA"}
        present_subsets = np.unique(subset_subsets)
        
        for subset in present_subsets:
            subset_data = out_dataset.filter(lambda x: x["subset"] == subset)
            accuracy = sum(subset_data["results"]) / len(subset_data["results"])
            results_grouped[subset] = accuracy
            print(f"{subset}: {accuracy:.4f}")

        if not args.pref_sets:
            section_scores = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
            print("\nSection Scores:", section_scores)

        if not args.do_not_save:
            save_to_hub(
                results_grouped,
                args.model + "_lora",
                "eval-set/",
                args.debug,
                save_metrics_for_beaker=not args.disable_beaker_save,
            )

if __name__ == "__main__":
    # Enable multiprocessing for DataLoader
    mp.set_start_method('spawn', force=True)
    main()