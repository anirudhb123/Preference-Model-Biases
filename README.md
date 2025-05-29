# Counterfactual Data Augmentation (CDA) Pipeline

> **High-Level Overview**
> 
> For detailed pipeline usage and fine-tuning instructions, see [`main/README.md`](main/README.md).

## 🎯 Project Overview

The CDA Pipeline is a comprehensive toolkit for detecting and mitigating biases in language models through counterfactual data augmentation. It supports multiple bias types including length, vagueness, jargon, structure, and sycophancy biases.

## 📁 Repository Structure

```text
.
├── data/                     # Pipeline inputs and outputs
│   ├── baseline_responses/   # Raw model outputs
│   ├── perturbations/        # Counterfactual examples
│   ├── training_labels/      # Bias labels
│   └── fine_tuned_results/   # Model checkpoints & metrics
│
├── main/                     # Core pipeline implementation
│   ├── bash_scripts/        # Execution scripts
│   └── README.md            # Detailed instructions
│
├── human_annotation_data/    # Validation datasets
├── data_utils/              # Analysis utilities
└── llm_evaluation_data/     # LLM evaluation results
```

## 🚀 Quick Start

1. **Core Components**: Only `data/` and `main/` directories are required to run the pipeline
2. **Setup Instructions**: See [`main/README.md`](main/README.md) for detailed setup and usage
3. **Pre-computed Data**: Access our perturbations dataset on [Hugging Face](https://huggingface.co/datasets/abharadwaj123/preference-model-perturbations)

## 📊 Supported Bias Types

- **Length Bias**: Mitigate preferences for response length
- **Vagueness Bias**: Balance specific vs. general responses
- **Jargon Bias**: Control technical language usage
- **Structure Bias**: Address format preferences
- **Sycophancy Bias**: Reduce excessive agreeableness

## 📚 Additional Resources

- **Paper Artifacts**: `data_utils/`, `human_annotation_data/`, and `llm_evaluation_data/` contain supplementary materials for our research paper
- **Human Annotations**: Curated examples for each bias type in `human_annotation_data/`
- **Evaluation Data**: LLM-based evaluation results in `llm_evaluation_data/`

## 🔗 Related Links

- [Detailed Documentation](main/README.md)
- [Pre-computed Perturbations](https://huggingface.co/datasets/abharadwaj123/preference-model-perturbations)

---

**Note**: For installation, configuration, and step-by-step usage instructions, please refer to [`main/README.md`](main/README.md).