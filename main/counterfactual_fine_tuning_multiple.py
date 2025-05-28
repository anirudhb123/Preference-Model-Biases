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
_BIASES = flags.DEFINE_string(
    "biases", "", "Biases for tuning"
)
_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing training examples."
)
_SECOND_INPUT_PATH = flags.DEFINE_string(
    "second_input_path", "",
    f"Path to the second input file containing additional examples with {_BIASES} labels."
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
# -------------------------------- Datasets --------------------------------- #
def prepare_mixed_counterfactual_data(
    input_paths: str,
    tokenizer,
    val_split: float
) -> tuple[Dataset, Dataset]:
    """
    For each bias in --biases, sample exactly N=_EXAMPLES.value examples
    from its corresponding file (in the same order as biases),
    then split each bias’s N into train/val (val_split fraction per bias)
    and concatenate across biases. Returns (train_ds, val_ds).
    """
    biases = [b.strip() for b in _BIASES.value.split(",") if b.strip()]
    paths = [p.strip() for p in input_paths.split(",") if p.strip()]
    if len(paths) != len(biases):
        raise ValueError(f"Need one path per bias ({len(biases)}), got {len(paths)}")

    def load_examples(path):
        exs = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    exs.append(json.loads(line))
        return exs

    N = _EXAMPLES.value
    k_val = int(N * val_split)
    train_examples, val_examples = [], []
    per_bias_counts = {}

    for bias, path in zip(biases, paths):
        pool = load_examples(path)
        if len(pool) < N:
            raise ValueError(f"Not enough examples in {path} for bias={bias}: got {len(pool)}, need {N}")
        sampled = random.sample(pool, N)
        random.shuffle(sampled)

        val_part   = sampled[:k_val]
        train_part = sampled[k_val:]

        per_bias_counts[bias] = {"train": len(train_part), "val": len(val_part)}
        train_examples.extend(train_part)
        val_examples.extend(val_part)

    # summary
    total_train = sum(c["train"] for c in per_bias_counts.values())
    total_val   = sum(c["val"]   for c in per_bias_counts.values())
    print(">> Sampling summary:")
    for b, cnts in per_bias_counts.items():
        print(f"   • {b}: train={cnts['train']} | val={cnts['val']}")
    print(f"   → TOTAL: train={total_train} | val={total_val}")

    # tokenization helper
    def tokenize_batch(batch):
        # build your per‐example conversations
        convs_chosen = [
            [
                {"role": "user", "content": q},
                {"role": "assistant", "content": cr},
            ]
            for q, cr in zip(batch["query"], batch["chosen_response"])
        ]
        convs_rejected = [
            [
                {"role": "user", "content": q},
                {"role": "assistant", "content": rr},
            ]
            for q, rr in zip(batch["query"], batch["rejected_response"])
        ]

        # apply_chat_template over the whole batch, but tell it not to tokenize
        texts_chosen  = tokenizer.apply_chat_template(convs_chosen,  tokenize=False)
        texts_rejected = tokenizer.apply_chat_template(convs_rejected, tokenize=False)

        # now texts_* are List[str], so this will succeed
        chosen_enc = tokenizer(
            texts_chosen,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        rejected_enc = tokenizer(
            texts_rejected,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # return plain lists for HF Datasets
        return {
            "input_ids_chosen":        chosen_enc.input_ids.tolist(),
            "attention_mask_chosen":   chosen_enc.attention_mask.tolist(),
            "input_ids_rejected":      rejected_enc.input_ids.tolist(),
            "attention_mask_rejected": rejected_enc.attention_mask.tolist(),
        }


    # build HF datasets
    def make_ds(ex_list):
        return Dataset.from_dict({
            "query":             [e["query"] for e in ex_list],
            "chosen_response":   [e["chosen_response"] for e in ex_list],
            "rejected_response": [e["rejected_response"] for e in ex_list],
        }).map(
            tokenize_batch,
            batched=True, batch_size=4,
            remove_columns=["query", "chosen_response", "rejected_response"],
        )

    return make_ds(train_examples), make_ds(val_examples)

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
    train_dataset, val_dataset = prepare_mixed_counterfactual_data(
        _INPUT_PATH.value,          # now a comma-separated list of all your per-bias files
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
