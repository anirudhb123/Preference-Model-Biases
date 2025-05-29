# CDA-Pipeline Overview

> **High-Level Repo Guide**
> For detailed pipeline usage and fine‐tuning instructions, see **`main/README.md`**.

This repository contains the full Counterfactual Data Augmentation (CDA) post‑training ecosystem:

```text
.
├── data/                     # Input prompts & all generated outputs
│   ├── baseline_responses/   # Raw model outputs on your prompts
│   ├── perturbations/        # Counterfactually modified examples
│   ├── training_labels/      # Human or synthetic labels
│   └── fine_tuned_results/   # Trained checkpoints & evaluation metrics

├── main/                     # Core pipeline code
│   └── README.md             # Detailed end‑to‑end instructions + config

├── data_utils/               # Paper artifacts: plotting & figure scripts
├── human_annotation_data/    # Paper artifacts: raw human judgments
└── llm_evaluation_data/      # Paper artifacts: LLM‑based evaluation data
```

---

**Key points:**

* **`data/`** and **`main/`** are the *only* folders needed to run or adapt the pipeline.
* Everything else under the root (e.g. `data_utils/`, `human_annotation_data/`, `llm_evaluation_data/`) supports figures, analysis, and results for our accompanying paper.
* For installation, configuration, and step‑by‑step commands, check **`main/README.md`**.