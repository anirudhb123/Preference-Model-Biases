#!/usr/bin/env python3
"""Fine-tune a preference model on new examples and push to Hugging Face Hub."""

from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model
from huggingface_hub import snapshot_download

import logging
import traceback
import time
import os
import sys
import torch
import wandb  # Added wandb import

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from absl import app
from absl import flags
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import jsonl_utils
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoConfig,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import random
import numpy as np
import evaluate
import tempfile
from torch.utils.data import DataLoader
from typing import Dict, Union, Any
import json

# --------------------------------------------------------------------------- #
#                                 Flag setup                                  #
# --------------------------------------------------------------------------- #
_BIAS = flags.DEFINE_string(
    "bias", "", "Bias for tuning"
)
_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing training examples."
)
_SECOND_INPUT_PATH = flags.DEFINE_string(
    "second_input_path", "",
    f"Path to the second input file containing additional examples with {_BIAS} labels."
)
_MODEL_REPO_ID = flags.DEFINE_string(
    "model_repo_id", "", "Hugging Face Hub repository ID (e.g., 'username/model-name')."
)
_HF_TOKEN = flags.DEFINE_string(
    "hf_token", "", "Hugging Face API token. If not provided, will look for HUGGING_FACE_HUB_TOKEN env var."
)
_BASE_MODEL_NAME = flags.DEFINE_string(
    "base_model_name", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2", "Base model name."
)
_EPOCHS = flags.DEFINE_integer(
    "epochs", 3, "Number of training epochs."
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 8, "Training batch size."
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 2e-5, "Learning rate for fine-tuning."
)
_USE_LORA = flags.DEFINE_boolean(
    "use_lora", True, "Whether to use LoRA for efficient fine-tuning."
)
_LORA_R = flags.DEFINE_integer(
    "lora_r", 16, "Rank for LoRA adapter."
)
_LORA_ALPHA = flags.DEFINE_integer(
    "lora_alpha", 32, "Alpha for LoRA adapter."
)
_VALIDATION_SPLIT = flags.DEFINE_float(
    "validation_split", 0.1, "Fraction of data to use for validation."
)
_SEED = flags.DEFINE_integer(
    "seed", 42, "Random seed for reproducibility."
)
_EXAMPLES = flags.DEFINE_integer(
    "examples", 1512, "Number of examples to use from the first input file."
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project", "preference-model-finetuning", "Weights & Biases project name."
)
_WANDB_RUN_NAME = flags.DEFINE_string(
    "wandb_run_name", "", "Weights & Biases run name. If not provided, a default will be generated."
)
_DISABLE_WANDB = flags.DEFINE_boolean(
    "disable_wandb", False, "Disable Weights & Biases logging."
)
_EVAL_INTERVAL = flags.DEFINE_integer(
    "eval_interval", 20, "Evaluate the model every N training steps."
)

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #

def login_to_hub(token: str | None):
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("❗ No HF token — skipping login")

# --------------------------------------------------------------------------- #
#                                RewardTrainer                                #
# --------------------------------------------------------------------------- #
class RewardTrainer:
    def __init__(
        self, model, args, train_dataset, eval_dataset,
        tokenizer, data_collator, compute_metrics
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.best_accuracy = 0.0

        # Initialise wandb run here if not disabled.
        if not _DISABLE_WANDB.value:
            run_name = (
                _WANDB_RUN_NAME.value
                if _WANDB_RUN_NAME.value
                else f"{_MODEL_REPO_ID.value.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            config = {
                "model_name": _BASE_MODEL_NAME.value,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "epochs": args.num_train_epochs,
                "use_lora": _USE_LORA.value,
                "lora_r": _LORA_R.value if _USE_LORA.value else None,
                "lora_alpha": _LORA_ALPHA.value if _USE_LORA.value else None,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
            }
            wandb.init(project=_WANDB_PROJECT.value, name=run_name, config=config)
            wandb.watch(model, log="all", log_freq=50)

    # ------------------------------ Training loop --------------------------- #
    def train(self):
        print("🚀 Entering custom training method")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate
        )

        global_step = 0
        log_interval = 10  # Log every 10 batches

        for epoch in range(self.args.num_train_epochs):
            print(f"🌟 Epoch {epoch+1}/{self.args.num_train_epochs}")
            self.model.train()
            total_loss = 0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                t1 = time.time()
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                scores = outputs.logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(
                    scores[::2] - scores[1::2]
                ).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()

                batch_loss = loss.item()
                total_loss += batch_loss
                global_step += 1

                print(
                    f"[Batch {batch_idx}] Compute: {time.time()-t1:.2f}s "
                    f"| Loss={batch_loss:.4f}",
                    flush=True,
                )

                if not _DISABLE_WANDB.value and batch_idx % log_interval == 0:
                    wandb.log(
                        {
                            "train/batch_loss": batch_loss,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/compute_time": time.time() - t1,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

                # Evaluate every _EVAL_INTERVAL training steps
                if (
                    global_step % _EVAL_INTERVAL.value == 0
                    and self.eval_dataset is not None
                ):
                    eval_results = self.evaluate()
                    accuracy = eval_results["accuracy"]
                    val_loss = eval_results.get("loss", 0.0)
                    print(
                        f"[Step {global_step}] Validation Accuracy: {accuracy:.4f} "
                        f"- Validation Loss: {val_loss:.4f}"
                    )

                    if not _DISABLE_WANDB.value:
                        wandb.log(
                            {
                                "eval/accuracy": accuracy,
                                "eval/loss": val_loss,
                                "eval/step": global_step,
                            },
                            step=global_step,
                        )

                    if accuracy > self.best_accuracy:
                        print(
                            f"✅ New best accuracy "
                            f"({accuracy:.4f} > {self.best_accuracy:.4f}), "
                            "saving model locally"
                        )
                        self.best_accuracy = accuracy
                        self.save_model(save_to_hub=False)
                        if not _DISABLE_WANDB.value:
                            wandb.log(
                                {
                                    "best/accuracy": accuracy,
                                    "best/loss": val_loss,
                                    "best/step": global_step,
                                },
                                step=global_step,
                            )
                            wandb.save(os.path.join(self.args.output_dir, "*"))

                    self.model.train()

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s"
            )

            if not _DISABLE_WANDB.value:
                wandb.log(
                    {
                        "train/epoch": epoch + 1,
                        "train/epoch_loss": avg_loss,
                        "train/epoch_time": epoch_time,
                    },
                    step=global_step,
                )

    # ------------------------------ Evaluation ------------------------------ #
    def evaluate(self):
        self.model.eval()
        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )
        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.squeeze(-1)

                # Calculate loss
                scores = logits.reshape(-1, 2)
                loss = -torch.nn.functional.logsigmoid(
                    scores[:, 0] - scores[:, 1]
                ).mean().item()
                total_loss += loss

                # Calculate accuracy
                preds = scores[:, 0] > scores[:, 1]
                correct += preds.sum().item()
                total += preds.numel()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return {"accuracy": accuracy, "loss": avg_loss}

    # ----------------------------- Save checkpoint -------------------------- #
    def save_model(self, save_to_hub: bool = True):
        self.model.save_pretrained(self.args.output_dir)

        if save_to_hub:
            self.model.push_to_hub(self.args.hub_model_id)

        # Save model artifacts to wandb if enabled.
        if not _DISABLE_WANDB.value:
            artifact_dir = os.path.join(self.args.output_dir, "wandb_artifacts")
            os.makedirs(artifact_dir, exist_ok=True)

            model_config = {
                "model_name": _BASE_MODEL_NAME.value,
                "best_accuracy": self.best_accuracy,
                "use_lora": _USE_LORA.value,
                "lora_r": _LORA_R.value if _USE_LORA.value else None,
                "lora_alpha": _LORA_ALPHA.value if _USE_LORA.value else None,
            }
            with open(os.path.join(artifact_dir, "model_config.json"), "w") as f:
                json.dump(model_config, f)

            model_artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description=f"Fine-tuned reward model with accuracy {self.best_accuracy:.4f}",
            )
            model_artifact.add_dir(artifact_dir)
            wandb.log_artifact(model_artifact)

# --------------------------------------------------------------------------- #
#                           Data processing helpers                           #
# --------------------------------------------------------------------------- #
def pairwise_data_collator(features: list) -> Dict[str, torch.Tensor]:
    """Collate paired examples (chosen vs rejected)."""
    batch = {"input_ids": [], "attention_mask": []}
    for feature in features:
        batch["input_ids"].append(torch.tensor(feature["input_ids_chosen"]))
        batch["input_ids"].append(torch.tensor(feature["input_ids_rejected"]))
        batch["attention_mask"].append(torch.tensor(feature["attention_mask_chosen"]))
        batch["attention_mask"].append(torch.tensor(feature["attention_mask_rejected"]))
    return {
        "input_ids": torch.stack(batch["input_ids"]),
        "attention_mask": torch.stack(batch["attention_mask"]),
    }

# -------------------------------- Datasets --------------------------------- #
def prepare_counterfactual_data(
    input_path, second_input_path, tokenizer, val_split=0.1
):
    """
    Prepare dataset with correct pairing structure.

    This function:
      • Loads _EXAMPLES.value examples from the first input file.
      • Loads (1000 - _EXAMPLES.value) examples from the second input file,
        sampled in the following proportions:
           Yes/Yes: 46.9 %
           Yes/No : 19.2 %
           No/No : 34.0 %
    """

    # Desired sampling proportions for each bias type, taken from labeled training data sample (length is only 1000 longer examples)
    bias_proportions = {
        "structure": [2113, 864, 1532],  # [Yes/Yes, Yes/No, No/No]
        "jargon": [765, 446, 374],  # [Yes/Yes, Yes/No, No/No]
        "hedging": [806, 256, 1005], #  # [Yes/Yes, Yes/No, No/No]
        "vagueness": [196, 136, 755] #  # [Yes/Yes, Yes/No, No/No]
    }

    # ----------------- First input file ----------------- #
    examples_first: list[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8-sig") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue  # skip blank lines
            try:
                examples_first.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error in {input_path} on line {idx}: {e}")
                print(">>>", repr(line))
                raise

    random.shuffle(examples_first)
    examples_first = examples_first[: _EXAMPLES.value]

    # ---------------- Second input file ----------------- #
    examples_second_selected: list[Dict[str, Any]] = []
    if second_input_path:
        examples_second = []
        with open(second_input_path, "r", encoding="utf-8-sig") as f2:
            for idx, raw in enumerate(f2, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    examples_second.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"❌ JSON parse error in {second_input_path} on line {idx}: {e}"
                    )
                    print(">>>", repr(line))
                    raise

        # Bucket by flag combinations
        cat_yes_yes, cat_yes_no, cat_no_no = [], [], []
        for ex in examples_second:
            chosen_flag = ex.get(f"chosen_{_BIAS.value}", "No")
            rejected_flag = ex.get(f"rejected_{_BIAS.value}", "No")
            if chosen_flag == "Yes" and rejected_flag == "Yes":
                cat_yes_yes.append(ex)
            elif chosen_flag == "Yes" and rejected_flag == "No":
                cat_yes_no.append(ex)
            elif chosen_flag == "No" and rejected_flag == "No":
                cat_no_no.append(ex)

        # How many examples do we still need?
        num_second = max(0, 1000 - _EXAMPLES.value)

        total_count = sum(bias_proportions[_BIAS.value])
        prop_yy, prop_yn, prop_nn = [
            p / total_count for p in bias_proportions[_BIAS.value]
        ]

        num_yy = int(round(prop_yy * num_second))
        num_yn = int(round(prop_yn * num_second))
        num_nn = num_second - num_yy - num_yn  # ensure total matches

        print("NUMBERS NEEDED")
        print(num_yy, num_yn, num_nn, sep="\n")
        print("Numbers available")
        print(len(cat_yes_yes), len(cat_yes_no), len(cat_no_no), sep="\n")

        random.shuffle(cat_yes_yes)
        random.shuffle(cat_yes_no)
        random.shuffle(cat_no_no)
        examples_second_selected = (
            cat_yes_yes[:num_yy] + cat_yes_no[:num_yn] + cat_no_no[:num_nn]
        )

    # ---------------- Combine & split ------------------ #
    combined_examples = examples_first + examples_second_selected
    random.shuffle(combined_examples)

    print("First Set", len(examples_first))
    print("Second Set", len(examples_second_selected))
    print("LENGTH OF COMBINED EXAMPLES!", len(combined_examples))

    split_idx = int(len(combined_examples) * (1 - val_split))
    train_examples = combined_examples[:split_idx]
    val_examples = combined_examples[split_idx:]

    # --------------- Tokenisation helper --------------- #
    def process_batch(batch):
        chosen = tokenizer.apply_chat_template(
            [
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": cr},
                ]
                for q, cr in zip(batch["query"], batch["chosen_response"])
            ],
            tokenize=False,
        )
        rejected = tokenizer.apply_chat_template(
            [
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": rr},
                ]
                for q, rr in zip(batch["query"], batch["rejected_response"])
            ],
            tokenize=False,
        )

        chosen_enc = tokenizer(
            chosen,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        rejected_enc = tokenizer(
            rejected,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    # --------------- HuggingFace Datasets --------------- #
    train_dataset = Dataset.from_dict(
        {
            "query": [ex["query"] for ex in train_examples],
            "chosen_response": [ex["chosen_response"] for ex in train_examples],
            "rejected_response": [ex["rejected_response"] for ex in train_examples],
        }
    ).map(
        process_batch,
        batched=True,
        batch_size=4,
        remove_columns=["query", "chosen_response", "rejected_response"],
    )

    val_dataset = Dataset.from_dict(
        {
            "query": [ex["query"] for ex in val_examples],
            "chosen_response": [ex["chosen_response"] for ex in val_examples],
            "rejected_response": [ex["rejected_response"] for ex in val_examples],
        }
    ).map(
        process_batch,
        batched=True,
        batch_size=4,
        remove_columns=["query", "chosen_response", "rejected_response"],
    )

    return train_dataset, val_dataset

# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main(argv):
    # Seed everything
    random.seed(_SEED.value)
    np.random.seed(_SEED.value)
    torch.manual_seed(_SEED.value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED.value)

    # Hugging Face & W&B login
    login_to_hub(os.environ.get("HF_TOKEN"))
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    print(f"Loading base model: {_BASE_MODEL_NAME.value}")

    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_NAME.value)
    tokenizer.pad_token = tokenizer.eos_token  # Critical fix

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL_NAME.value,
        quantization_config=bnb_config,
        device_map="cuda",
        num_labels=1,
    )

    if _USE_LORA.value:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=_LORA_R.value,
            lora_alpha=_LORA_ALPHA.value,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare datasets
    train_dataset, val_dataset = prepare_counterfactual_data(
        _INPUT_PATH.value,
        _SECOND_INPUT_PATH.value,
        tokenizer,
        val_split=_VALIDATION_SPLIT.value,
    )

    training_args = TrainingArguments(
        gradient_checkpointing=False,
        output_dir=_MODEL_REPO_ID.value,
        learning_rate=_LEARNING_RATE.value,
        per_device_train_batch_size=_BATCH_SIZE.value,
        per_device_eval_batch_size=_BATCH_SIZE.value,
        num_train_epochs=_EPOCHS.value,
        evaluation_strategy="epoch",
        logging_steps=10,
        push_to_hub=True,
        hub_model_id=_MODEL_REPO_ID.value,
        hub_strategy="end",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=pairwise_data_collator,
        compute_metrics=lambda preds: {
            "accuracy": (np.array(preds[0][::2]) > np.array(preds[0][1::2])).mean()
        },
    )

    print("Starting training...")
    trainer.train()
    print("🔖 Pushing best model to Hugging Face Hub…")
    trainer.save_model(save_to_hub=True)
    try:
        if not _DISABLE_WANDB.value and wandb.run is not None:
            wandb.finish()
    except OSError as e:
        logger.warning(f"wandb.finish() encountered an OSError and will be ignored: {e}")

# --------------------------------------------------------------------------- #
#                                   Entrypoint                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(main)
