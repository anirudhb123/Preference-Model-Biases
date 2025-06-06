# Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models

## Overview

Preference models can exhibit idiosyncratic biases—favoring length, structure, jargon, vagueness, or sycophancy over true quality.  

**In this work we:**

- Construct controlled counterfactual response pairs for each bias  
- Measure misalignment with human judgments and quantify miscalibration  
- Fine-tune via targeted counterfactual data augmentation (CDA) to correct bias  

For detailed CDA pipeline usage and fine-tuning instructions, see [`main/README.md`](main/README.md).

## 📁 Repository Structure

```text
.
├── data/                                   # All pipeline inputs and outputs
│   ├── chatbot_arena_labeled_data/         # Counterfactually perturbed examples sampled from Chatbot Arena
│   ├── fine_tuned_model_scores/            # Reward-model scores on bias-augmented, fine-tuned data
│   ├── perturbations/                      # Generated bias perturbations for original model responses
│   ├── reward_model_counterfactual_data/   # Counterfactual examples used during reward-model fine-tuning
│   ├── reward_model_training_labeled_data/ # Human-annotated examples for training the reward model
│   └── rewardbench_results/                # Benchmark metrics and evaluation outputs on RewardBench
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

## Paper and citation

You can find the arXiv paper [here](https://arxiv.org/abs/2506.05339)

```bibtex
@article{bharadwaj2025flattery,
  title={Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models},
  author={Bharadwaj, Anirudh and Malaviya, Chaitanya and Joshi, Nitish and Yatskar, Mark},
  journal={arXiv preprint arXiv:2506.05339},
  year={2025},
  archivePrefix={arXiv},
  eprint={2506.05339},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.05339}
}
```
