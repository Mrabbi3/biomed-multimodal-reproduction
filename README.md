# Reproducing Med-PaLM M: An Open-Source Approach to Generalist Biomedical AI

[![Paper](https://img.shields.io/badge/Paper-arXiv%202307.14334-red)](https://arxiv.org/abs/2307.14334)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mrabbi3/biomed-multimodal-reproduction/blob/main/notebooks/Full_Reproduction_Colab.ipynb)

An independent reproduction study of [Med-PaLM M](https://arxiv.org/abs/2307.14334) (Tu et al., 2023) — the first generalist biomedical AI — using open-source models. We replace Google's proprietary PaLM-E (562B) with BLIP-2 (3.4B) and implement the paper's complete medical VQA pipeline: instruction prompting, one-shot exemplars, LoRA fine-tuning, and standardized evaluation.

**Author:** MD Rabbi · Department of Computer Science · February 2026

---

## Key Findings

| Model | Parameters | BLEU-1 (%) | F1 (%) | Source |
|-------|-----------|-----------|--------|--------|
| Prior SOTA (specialist) | Various | 71.03 | — | Paper Table 2 |
| PaLM-E 84B (zero-shot) | 84B | 59.19 | 38.67 | Paper Table 2 |
| Med-PaLM M 12B | 12B | 64.02 | 50.66 | Paper Table 2 |
| Med-PaLM M 562B | 562B | 71.27 | 62.06 | Paper Table 2 |
| **Ours: BLIP-2 (zero-shot)** | **3.4B** | **0.44** | **0.70** | **This work** |

**Exemplar Ablation:** One-shot prompting produces a **2.7x improvement** in BLEU-1 (0.95% → 2.54%), independently confirming the paper's prompting strategy works across model scales.

---

## What This Project Does

Med-PaLM M showed that a single AI model can handle radiology VQA, pathology analysis, report generation, and more — all at once. But it uses PaLM-E, a proprietary 562-billion-parameter model nobody outside Google can access.

This project asks: **can we reproduce the methodology with open-source tools?** We implement the full pipeline — from data loading to evaluation — and test it on VQA-RAD (3,515 radiology question-answer pairs). The 165x parameter gap means we won't match their absolute numbers, but we validate key methodological claims and provide a complete, runnable codebase for the research community.

---

## Quick Start

### Option 1: Google Colab (Recommended — No Setup Required)

Click the badge above or open `notebooks/Full_Reproduction_Colab.ipynb` in Colab. Select a **T4 GPU** runtime and run cells sequentially. The full pipeline completes in ~1 hour.

### Option 2: Local Installation

```bash
git clone https://github.com/Mrabbi3/biomed-multimodal-reproduction.git
cd biomed-multimodal-reproduction
pip install -r requirements.txt

# Download dataset
python data/download.py --dataset vqa_rad

# Run experiments sequentially
python experiments/01_data_sanity_check.py
python experiments/02_forward_pass_test.py --model blip2 --quantize
python experiments/03_overfit_single_batch.py --quantize
python experiments/04_train_vqa.py --epochs 10 --quantize
python experiments/05_zero_shot_eval.py --experiment all --quantize

# Generate comparison report
python evaluation/compare_to_paper.py
```

**Requirements:** Python 3.10+, NVIDIA GPU with 16GB+ VRAM, PyTorch 2.0+

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Med-PaLM M (Paper)                    │
│  Image → ViT → 256 tokens ─┐                           │
│                              ├→ PaLM-E (562B) → Answer  │
│  Instruction + Question ────┘                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Our Reproduction                        │
│  Image → ViT → Q-Former ───┐                           │
│                              ├→ Flan-T5-XL (3B) → Answer│
│  Instruction + Question ────┘    via BLIP-2             │
│                                  + LoRA adapters         │
└─────────────────────────────────────────────────────────┘
```

| Component | Med-PaLM M | Our Reproduction |
|-----------|-----------|------------------|
| Language model | PaLM-E (562B) | Flan-T5-XL (3B) via BLIP-2 |
| Vision-language bridge | Linear projection | Q-Former (32 learned queries) |
| Training | Full fine-tuning on TPU v4 | LoRA (rank=16) on single T4 GPU |
| Optimizer | Adafactor | Adafactor (same) |
| Prompting | One-shot exemplar | One-shot exemplar (same) |
| Compute time | Weeks | ~1 hour |

---

## Repository Structure

```
biomed-multimodal-reproduction/
├── models/                          # Model wrappers
│   ├── blip2_wrapper.py             #   BLIP-2 with Flan-T5 (primary)
│   ├── llava_med_wrapper.py         #   LLaVA-Med (alternative)
│   └── base_model.py               #   Abstract interface
├── data/                            # Data pipeline
│   ├── download.py                  #   Hugging Face dataset download
│   ├── vqa_rad_loader.py            #   VQA-RAD loader (3,515 QA pairs)
│   ├── slake_loader.py              #   Slake-VQA loader (14,028 QA pairs)
│   └── preprocessing.py             #   224x224 resize, grayscale to RGB
├── training/                        # Training pipeline
│   ├── trainer.py                   #   LoRA fine-tuning with Adafactor
│   ├── prompts.py                   #   All 9 instruction templates from paper
│   └── multitask_mixer.py           #   Proportional task sampling
├── evaluation/                      # Evaluation suite
│   ├── metrics.py                   #   BLEU-1, token F1 (paper Section A.3)
│   ├── evaluate.py                  #   Full test set evaluation
│   └── compare_to_paper.py          #   Table 2 comparison + charts
├── experiments/                     # 5-phase experiment pipeline
│   ├── 01_data_sanity_check.py      #   Verify data + preprocessing
│   ├── 02_forward_pass_test.py      #   Zero-shot baseline
│   ├── 03_overfit_single_batch.py   #   Training pipeline validation
│   ├── 04_train_vqa.py              #   Full training + evaluation
│   └── 05_zero_shot_eval.py         #   Generalization experiments
├── configs/                         # Hyperparameters from paper Tables A.1-A.2
├── tests/                           # 27 unit tests
├── notebooks/                       # Google Colab notebook
├── docs/                            # Setup guide, dataset access, reproduction log
└── results/                         # Generated metrics, plots, comparison tables
```

---

## Experiments

### Experiment 1: Data Sanity Check
Verifies dataset loading, image preprocessing (224x224 resize with padding), and metric correctness with synthetic inputs.

### Experiment 2: Zero-Shot Baseline
Evaluates BLIP-2 on VQA-RAD without fine-tuning. Comparable to the paper's PaLM-E 84B baseline. Our 3.4B model achieves 0.44% BLEU-1 vs their 84B model's 59.19%.

### Experiment 3: Overfit Test
Trains on 5 examples for 50 epochs to verify gradient flow, LoRA integration, and loss convergence before committing to full training.

### Experiment 4: Full Training
Fine-tunes BLIP-2 with LoRA on VQA-RAD training set (1,793 samples), evaluates on test set (451 samples), and generates side-by-side comparison with paper baselines.

### Experiment 5: Generalization
Tests the paper's key claims: one-shot exemplar ablation shows a 2.7x BLEU-1 improvement, confirming the prompting strategy transfers across model scales.

---

## Datasets

| Dataset | QA Pairs | Images | Modalities | Used For |
|---------|----------|--------|------------|----------|
| [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | 3,515 | 315 | CT, MRI, X-ray | Primary evaluation |
| [Slake-VQA](https://huggingface.co/datasets/mdwiratathya/SLAKE) | 14,028 | 642 | CT, MRI, X-ray | Cross-dataset transfer |
| [Path-VQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | 32,799 | 4,998 | Pathology | Future work |

---

## Citation

If you use this codebase in your research:

```bibtex
@misc{rabbi2026reproducing,
  title={Reproducing Med-PaLM M: An Open-Source Approach to Generalist Biomedical AI},
  author={Rabbi, MD},
  year={2026},
  url={https://github.com/Mrabbi3/biomed-multimodal-reproduction}
}
```

Original paper:
```bibtex
@article{tu2023towards,
  title={Towards Generalist Biomedical AI},
  author={Tu, Tao and others},
  journal={arXiv preprint arXiv:2307.14334},
  year={2023}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
