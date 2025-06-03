# CDA Post-Training Pipeline

--- 

![pipeline_overview](../data_utils/plots/github/pipeline_overview.gif)
--- 

This document provides step-by-step instructions for running the **Counterfactual Data Augmentation (CDA)** post-training pipeline developed to mitigate idiosyncratic biases in reward models. It leverages controlled counterfactual perturbations for length, structure, jargon, vagueness, and sycophancy to fine-tune models toward human-aligned judgments. For a high-level overview of our approach, see [`README.md`](../README.md)

---

## 📋 Table of Contents

- [Initial Setup](#-initial-setup)
- [Pipeline Overview](#-pipeline-overview)
- [Data Preparation](#-data-preparation)
- [Fine-Tuning](#-fine-tuning)
- [Evaluation](#-evaluation)
- [Human Annotation Data](#-human-annotation-data)
- [Supported Bias Types](#-supported-bias-types)
- [Bash Scripts Reference](#-bash-scripts-reference)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Initial Setup

### Prerequisites

- Python 3.8+
- Access to OpenAI API (for GPT models)
- Hugging Face account and token
- Weights & Biases account (optional, for logging)

### Installation

1. **Install dependencies**

   ```bash
   pip install -r main/requirements.txt
   ```

2. **Configure API keys**
   
   Create a script `main/bash_scripts/set_keys.sh` with your API credentials:

   ```bash
   #!/bin/bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export HF_API_TOKEN="your_huggingface_token_here"
   export WANDB_API_KEY="your_wandb_api_key_here"  # Optional
   ```

   Make it executable and source it before running any pipeline scripts:

   ```bash
   chmod +x main/bash_scripts/set_keys.sh
   source main/bash_scripts/set_keys.sh
   ```

### Directory Structure

The repository is organized as follows:

```
├── data/                           # Pipeline inputs and outputs
├── main/                          # Core pipeline scripts
│   ├── bash_scripts/              # Orchestration wrappers
│   └── *.py                       # Python modules
├── human_annotation_data/         # Human-annotated validation examples
│   ├── combined_human_annotations.jsonl
│   ├── group_human_annotations.py
│   ├── length/                    # Length bias annotations
│   ├── vagueness/                 # Vagueness bias annotations
│   ├── jargon/                    # Jargon bias annotations
│   ├── structure/                 # Structure bias annotations
│   └── sycophancy/                # Sycophancy bias annotations
└── README.md                      # This file
```

---

## 📊 Pipeline Overview

The CDA pipeline consists of three main phases:

1. **Data Preparation**: Generate and label biased examples
2. **Fine-Tuning**: Train models to mitigate identified biases
3. **Evaluation**: Assess model performance and bias reduction

The pipeline supports both single-bias and multi-bias mitigation approaches.

---

## 🔧 Data Preparation

### Step 1: Generate Baseline Responses

Generate initial responses from your base model:

```bash
bash main/bash_scripts/run_base.sh
```

**Output**: `data/baseline_responses/`

### Step 2: Generate Perturbations

Create biased variations of the baseline responses:

```bash
bash main/bash_scripts/run_perturbed.sh
```

**Alternative**: Use our pre-computed perturbations from Hugging Face:
- 📦 [Pre-computed Perturbations Dataset](https://huggingface.co/datasets/abharadwaj123/preference-model-perturbations)

**Output**: `data/perturbed_responses/`

### Step 3: Label Training Data

Identify which examples contain bias:

```bash
bash main/bash_scripts/run_data_labeling.sh
```

**Output**: `data/labeled_data/`

### Step 4: Generate Counterfactuals

Create counterfactual examples for fine-tuning:

```bash
bash main/bash_scripts/run_counterfactual_generation.sh
```

**Output**: `data/counterfactuals/`

### Step 5: (Optional) Sample Chatbot Arena Data

Incorporate real-world examples from Chatbot Arena:

```bash
bash main/bash_scripts/run_chatbot_labeling.sh
```

**Output**: `data/chatbot_arena_samples/`

---

## 🎯 Fine-Tuning

Choose your approach based on whether you want to mitigate single or multiple biases:

### Single Bias Mitigation

Fine-tune your model to address one specific bias type:

```bash
bash main/bash_scripts/run_fine_tuning.sh
```

**Input**: Data from steps 4 and (optionally) 5
**Output**: Fine-tuned model in `data/models/single_bias/`

### Multiple Bias Mitigation

Fine-tune your model to address multiple bias types simultaneously:

```bash
bash main/bash_scripts/run_fine_tuning_multiple.sh
```

**Input**: Data from steps 4 and (optionally) 5
**Output**: Fine-tuned model in `data/models/multiple_bias/`

---

## 📈 Evaluation

### RewardBench Evaluation

Evaluate your fine-tuned model against the RewardBench benchmark:

**Single bias model**:
```bash
bash main/bash_scripts/run_rewardbench_eval.sh
```

**Multiple bias model**:
```bash
bash main/bash_scripts/run_rewardbench_eval_multiple.sh
```

**Output**: Results saved to `data/rewardbench_results/`

### Bias-Specific Evaluation

Score your model on the perturbation examples to measure bias reduction:

**Single bias model**:
```bash
bash main/bash_scripts/run_fine_tuned_rm.sh
```

**Multiple bias model**:
```bash
bash main/bash_scripts/run_fine_tuned_rm_multiple.sh
```

**Output**: Bias scores saved to `data/bias_evaluation/`

---

## 🧪 Human Annotation Data

The repository includes human-annotated examples for validation and comparison:

### Available Datasets

| Dataset | Description | Location |
|---------|-------------|----------|
| **Combined** | All bias types together | `human_annotation_data/combined_human_annotations.jsonl` |
| **Length** | Length preference bias | `human_annotation_data/length/` |
| **Vagueness** | Specificity preference bias | `human_annotation_data/vagueness/` |
| **Jargon** | Technical language bias | `human_annotation_data/jargon/` |
| **Structure** | Format preference bias | `human_annotation_data/structure/` |
| **Sycophancy** | Agreement preference bias | `human_annotation_data/sycophancy/` |

### Processing Annotations

Use the provided script to group annotations by bias type:

```bash
python human_annotation_data/group_human_annotations.py
```

---

## 🔍 Supported Bias Types

This pipeline can detect and mitigate the following bias types:

| Bias Type | Description | Example |
|-----------|-------------|---------|
| **Length** | Preference for longer or shorter responses | Favoring verbose over concise answers |
| **Vagueness** | Preference for vague vs. specific responses | Preferring general over detailed explanations |
| **Jargon** | Preference for technical vs. accessible language | Favoring complex terminology over plain language |
| **Structure** | Preference for certain response formats | Preferring bullet points over paragraphs |
| **Sycophancy** | Preference for agreeable vs. truthful responses | Favoring responses that agree with user opinions |

---

## 📝 Bash Scripts Reference

### Core Pipeline Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run_base.sh` | Generate baseline responses | Queries | Baseline responses |
| `run_perturbed.sh` | Create biased perturbations | Baseline responses | Perturbed examples |
| `run_data_labeling.sh` | Label examples for bias | Perturbed examples | Labeled dataset |
| `run_counterfactual_generation.sh` | Generate training examples | Labeled dataset | Counterfactual pairs |
| `run_chatbot_labeling.sh` | Sample real-world examples | Chatbot Arena data | Labeled samples |

### Fine-Tuning Scripts

| Script | Purpose | Bias Scope |
|--------|---------|------------|
| `run_fine_tuning.sh` | Train single-bias model | One bias type |
| `run_fine_tuning_multiple.sh` | Train multi-bias model | Multiple bias types |

### Evaluation Scripts

| Script | Purpose | Model Type |
|--------|---------|------------|
| `run_rewardbench_eval.sh` | RewardBench evaluation | Single-bias model |
| `run_rewardbench_eval_multiple.sh` | RewardBench evaluation | Multi-bias model |
| `run_fine_tuned_rm.sh` | Bias-specific evaluation | Single-bias model |
| `run_fine_tuned_rm_multiple.sh` | Bias-specific evaluation | Multi-bias model |

---

## 🔧 Troubleshooting

### Common Issues

**API Key Errors**
- Ensure all required API keys are set in `set_keys.sh`
- Verify API keys are valid and have sufficient quota
- Check that the script is sourced before running pipeline commands

**Memory Issues**
- Reduce batch size in configuration files
- Use smaller model variants for testing
- Consider using gradient checkpointing for large models

**Data Processing Errors**
- Verify input data format matches expected schema
- Check file permissions in data directories
- Ensure sufficient disk space for intermediate outputs

**Model Training Issues**
- Monitor training logs for convergence issues
- Adjust learning rate and other hyperparameters
- Verify training data quality and balance

### Getting Help

- Check the root `README.md` for high-level guidance
- Review individual script documentation
- Examine output logs for detailed error messages
- Ensure all dependencies are correctly installed
