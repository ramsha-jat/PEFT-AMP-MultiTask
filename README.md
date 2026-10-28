# Parameter-Efficient Adaptation of Protein Language Models for Dual-Target Antimicrobial Peptide Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract & Contribution

Translating antimicrobial peptide (AMP) candidates into clinical therapeutics requires co-optimizing antibacterial potency while preventing off-target mammalian cytotoxicity (hemolysis). Conventional methodologies isolate these predictive tasks into disconnected models or experience steep performance decay under stringent sequence homology splits.

This work introduces an end-to-end multi-task parameter-efficient transfer learning architecture built on top of the **ESM-2 (150M)** foundational protein language model:
- **Low-Rank Adaptation (LoRA):** Injected directly into attention projection layers ($r=8, \alpha=16$), updating merely **$\sim 0.70\%$ of parameters** ($\sim 1.05\text{M}$ tunable weights) to eliminate catastrophic forgetting while preserving sequence-level structural representations.
- **Adaptive Multi-Task Loss Balancing:** Reconciles conflicting gradient dynamics between bacterial potency ($p\text{MIC}$) and hemolytic cytotoxicity ($\log_{10}\text{HC}_{50}$) via **Kendall homoscedastic uncertainty weighting**.
- **Homology-Controlled Validation:** Benchmarked across strict **QMAP splits (60% sequence identity threshold)** to guarantee out-of-distribution generalization without data leakage.
- **In Silico De-Novo Screening:** Deployed on a curated library of 749 newly designed peptides to isolate selective leads with Therapeutic Indices ($\text{TI} = \text{HC}_{50} / \text{MIC}$) up to **144.03**.

---

## Benchmark Results

Evaluated on the predefined **QMAP benchmark homology test splits**:

| Model Framework | Tuning Strategy | Tunable Params | *E. coli* $\rho$ | *E. coli* $r$ | Hemolysis $\rho$ | Hemolysis $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| QMAP HemoLinear | Linear Ridge | Descriptors (Fixed) | 0.35 | 0.40 | 0.15--0.25 | 0.18--0.28 |
| QMAP BERT-AmPEP60 | Full Fine-Tuning | 100% | 0.44 | 0.48 | N/A | N/A |
| Frozen ESM-2 (150M) | Linear Probing | 0% (Head Only) | 0.5780 | 0.6418 | 0.25 | 0.30 |
| Single-Task LoRA ($r=8$) | Dedicated LoRA | $2 \times 921\text{k}$ (0.61%) | **0.6966** | **0.7528** | 0.5314 | 0.5123 |
| Joint Multi-Task LoRA | Static Loss ($\lambda=0.5$) | 1.05M (0.70%) | 0.6251 | 0.6892 | 0.3981 | 0.4202 |
| **Joint Multi-Task LoRA (Ours)** | **Uncertainty Weighting** | **1.05M (0.70%)** | **0.6132** | **0.6816** | **0.5374** | **0.5541** |

*Our uncertainty-weighted joint framework significantly improves hemolytic toxicity rank correlation over standard linear probes and static loss baselines while maintaining high antibacterial predictive fidelity.*

---

## Repository Structure

```text
├── multitask_dataset_final.csv       # Preprocessed QMAP dataset with homology-controlled splits
├── hemolysis_dataset.csv             # Mammalian hemolysis benchmark evaluation subset
├── screened_denovo_leads.csv         # 749 screened de-novo peptide candidates and predicted TIs
├── train_multitask_uncertainty.py    # Main training script (ESM-2 + LoRA + Uncertainty Weighting)
├── train_multitask_lora.py           # Multi-task baseline with static loss weighting
├── train_ec_lora.py                  # Single-task E. coli potency baseline
├── train_hemo_lora.py                # Single-task mammalian hemolysis baseline
├── train_ec_frozen_baseline.py       # Frozen ESM-2 linear probe baseline
├── screen_peptides.py                # Screening pipeline for computing TI and prioritizing leads
├── generate_and_screen.py            # De-novo sequence generation and batch screening
├── make_fig1.py                      # Pipeline architecture and workflow diagram generator
├── make_fig2.py                      # Parity plots generator (Experimental vs Predicted)
└── README.md```
```

## Setup and Environment Installation

```git clone [https://github.com/ramsha-jat/PEFT-AMP-MultiTask.git](https://github.com/ramsha-jat/PEFT-AMP-MultiTask.git)
cd PEFT-AMP-MultiTask```


# 1. Create and activate a virtual environment
```python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate```

# 2. Install PyTorch with CUDA support
```pip install --upgrade pip
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)```

# 3. Install model fine-tuning and evaluation libraries
```pip install transformers peft datasets pandas numpy scikit-learn scipy matplotlib seaborn```
