# CDA Post-Training Pipeline

> **Detailed End-to-End Instructions**
> For a high-level overview, see the root **`README.md`**.

---

## 🚀 Initial Setup

1. **Install dependencies**

   ```bash
   pip install -r main/requirements.txt
   ```

2. **Define API keys**
   Create a script `main/bash_scripts/set_keys.sh` and export the following:

   ```bash
   export OPENAI_API_KEY="<your_openai_key>"
   export HF_API_TOKEN="<your_hf_token>"
   export WANDB_API_KEY="<your_wandb_key>"
   ```

   Then source it before running any pipeline scripts:

   ```bash
   source main/bash_scripts/set_keys.sh
   ```

3. **Directory structure**

   * **`data/`**: inputs & outputs from each step
   * **`main/`**: core scripts
   * **`main/bash_scripts/`**: orchestration wrappers

---


## Pipeline

### Data Preparation

1. Generate baseline responses given queries (```main/bash_scripts/run_base.sh```)
2. Generate perturbations for bias (```main/bash_scripts/run_perturbed.sh```)
3. Label training data subset with occurrences of bias (```main/bash_scripts/run_data_labeling.sh```)
4. Generate counterfactuals from this subset as examples for fine tuning (```main/bash_scripts/run_counterfactual_generation.sh```)
5. (OPTIONAL) In the proportion they appear in the labeled training data subset, sample and label examples from chatbot arena (```main/bash_scripts/run_chatblot_labeling.sh```)

### Fine-Tuning

#### Mitigating Single Bias

1. Fine tune your choice of model using examples collected in steps 4 and (optionally) 5 (```main/bash_scripts/run_fine_tuning.sh```)

#### Mitigating Multiple Biases

1. Fine tune your choice of model using examples collected in steps 1 and (optionally) 2 (```main/bash_scripts/run_fine_tuning_multiple.sh```)


### Evaluation of Fine-Tuned Model

1. Rewardbench evaluation (```main/bash_scripts/run_rewardbench_eval.sh``` or ```main/bash_scripts/run_rewardbench_eval_multiple.sh```)
    
    a. Check output file (outputted to `data/rewardbench_results`) for results

2. Score perturbed examples (```main/bash_scripts/run_fine_tuned_rm.sh``` or ```main/bash_scripts/run_fine_tuned_rm_multiple.sh```) 


## 📝 Bash Scripts Reference


Brief overview of each script in this directory:

| Script                                           | Description                                                                                                                 |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `main/bash_scripts/run_base.sh`                  | Generate baseline responses given queries.                                                                                                      |
| `main/bash_scripts/run_chatbot_labeling.sh`      | Selects examples from Chatbot Arena for fine-tuning (with counterfactual examples).                                         |
| `main/bash_scripts/run_counterfactual_generation.sh` | Generates counterfactual examples to probe for bias.                                                                     |
| `main/bash_scripts/run_data_labeling.sh`         | Labels training examples for the presence of bias.                                                                          |
| `main/bash_scripts/run_fine_tuning.sh`           | Fine-tunes the reward model on generated counterfactuals.                                                                   |
| `main/bash_scripts/run_fine_tuned_rm.sh`         | Scores perturbed inputs using the fine-tuned reward model.                                                                  |
| `main/bash_scripts/run_perturbed.sh`             | Generates perturbations for a particular bias (modify prompt in `generate_perturbed_responses.py` if using RATE).          |
| `main/bash_scripts/run_rewardbench_eval.sh`      | Computes evaluation metrics on the RewardBench benchmark.                                                                   |
| `main/bash_scripts/run_rewardbench_eval_multiple.sh` | Computes evaluation metrics on the RewardBench benchmark for model mitigating multiple biases.                                 |
| `main/bash_scripts/run_fine_tuned_rm_multiple.sh`   | Scores perturbed inputs using multiple fine-tuned reward model variants.                                                  |
| `main/bash_scripts/run_fine_tuning_multiple.sh`     | Fine-tunes the reward model across multiple bias settings generated counterfactuals across multiple biases.                                                   |
