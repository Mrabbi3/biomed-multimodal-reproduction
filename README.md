# 🧬 Biomedical Multimodal AI Reproduction

**Reproducing and extending findings from "Towards Generalist Biomedical AI" (Med-PaLM M)**

[![Paper](https://img.shields.io/badge/Paper-arXiv%202307.14334-red)](https://arxiv.org/abs/2307.14334)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)]()

---

## Overview

This project reproduces key experiments from Google's **Med-PaLM Multimodal (Med-PaLM M)** paper — the first demonstration of a generalist biomedical AI system capable of interpreting clinical language, medical imaging, and genomics with a single set of model weights.

Since Med-PaLM M's architecture (PaLM-E) is not open-source, this reproduction leverages open-source multimodal models (LLaVA-Med, BLIP-2) to replicate the paper's core findings on publicly available datasets from **MultiMedBench**.

### Research Questions

1. **Can open-source multimodal models match Med-PaLM M's performance** on Medical Visual Question Answering (VQA) tasks?
2. **Does the one-shot exemplar prompting strategy** described in the paper improve performance in open-source settings?
3. **Is there evidence of zero-shot generalization** to unseen medical concepts when fine-tuning on multiple biomedical tasks?

---

## Paper Summary

| Aspect | Details |
|--------|---------|
| **Model** | Med-PaLM M — built on PaLM-E (PaLM LLM + ViT vision encoder) |
| **Scales** | 12B, 84B, 562B parameters |
| **Benchmark** | MultiMedBench — 14 tasks, 12 datasets, 1M+ samples |
| **Modalities** | Clinical text, radiology, pathology, dermatology, mammography, genomics |
| **Key Result** | Single model matches or exceeds specialist SOTA on all 14 tasks |
| **Clinical Eval** | Radiologists preferred Med-PaLM M reports over human reports in up to 40.5% of cases |

### Architecture at a Glance

```
Input Image → ViT Encoder → 256 Visual Tokens ─┐
                                                 ├→ PaLM Language Model → Generated Text
Task Instruction + Context → Text Tokens ────────┘
```

The model uses **instruction task prompting** with a **text-only one-shot exemplar** — providing an example input-output pair where the image is replaced with a `<img>` placeholder. This preserves compute efficiency while conditioning the model's output format.

---

## Project Structure

```
biomed-multimodal-reproduction/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .env.example                 # Environment variable template
│
├── configs/                     # Experiment configurations
│   ├── vqa_rad.yaml
│   ├── slake_vqa.yaml
│   └── path_vqa.yaml
│
├── data/                        # Data loading & preprocessing
│   ├── __init__.py
│   ├── download.py              # Dataset download scripts
│   ├── vqa_rad_loader.py        # VQA-RAD dataset loader
│   ├── slake_loader.py          # Slake-VQA dataset loader
│   └── preprocessing.py         # Image resize, normalization
│
├── models/                      # Model wrappers & adapters
│   ├── __init__.py
│   ├── base_model.py            # Abstract base class
│   ├── llava_med_wrapper.py     # LLaVA-Med integration
│   └── blip2_wrapper.py         # BLIP-2 integration
│
├── training/                    # Fine-tuning pipeline
│   ├── __init__.py
│   ├── trainer.py               # Training loop
│   ├── prompts.py               # Instruction templates (from paper)
│   └── multitask_mixer.py       # Task mixture sampling
│
├── evaluation/                  # Metrics & evaluation
│   ├── __init__.py
│   ├── metrics.py               # BLEU, ROUGE-L, F1 implementations
│   ├── evaluate.py              # Full evaluation pipeline
│   └── compare_to_paper.py      # Side-by-side comparison with Table 2
│
├── experiments/                 # Experiment scripts
│   ├── 01_data_sanity_check.py  # Phase 1: Verify data loading
│   ├── 02_forward_pass_test.py  # Phase 2: Test model inference
│   ├── 03_overfit_single_batch.py # Phase 3: Memorization test
│   ├── 04_train_vqa.py          # Phase 4: Full training run
│   └── 05_zero_shot_eval.py     # Phase 5: Zero-shot generalization
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_explore_datasets.ipynb
│   ├── 02_model_playground.ipynb
│   └── 03_results_analysis.ipynb
│
├── results/                     # Experiment outputs
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── tests/                       # Unit tests
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   └── test_model_forward.py
│
└── docs/                        # Additional documentation
    ├── SETUP_GUIDE.md
    ├── DATASET_ACCESS.md
    └── REPRODUCTION_LOG.md
```

---

## Research Phases & Milestones

### Phase 1: Project Definition & Data Setup *(Weeks 1–2)*

| Task | Status | Deliverable |
|------|--------|-------------|
| Select target task(s): Medical VQA | ⬜ | Decision documented in `REPRODUCTION_LOG.md` |
| Download VQA-RAD dataset | ⬜ | `data/vqa_rad/` populated |
| Download Slake-VQA dataset | ⬜ | `data/slake/` populated |
| Build data loaders | ⬜ | `test_data_loader.py` passes |
| Run `01_data_sanity_check.py` | ⬜ | Verified: images match questions |

**Key Insight from Paper:** Images are resized to 224×224×3 with aspect ratio preserved via padding. Grayscale images are stacked to 3 channels.

### Phase 2: Model Selection & Baseline *(Weeks 3–4)*

| Task | Status | Deliverable |
|------|--------|-------------|
| Clone LLaVA-Med or BLIP-2 repo | ⬜ | Working model inference |
| Implement instruction prompting | ⬜ | `training/prompts.py` matches Figure 2 |
| Run `02_forward_pass_test.py` | ⬜ | Model produces text output |
| Establish baseline metrics (no fine-tuning) | ⬜ | Baseline numbers logged |

**Key Insight from Paper:** The one-shot exemplar uses a dummy `<img>` text placeholder instead of an actual image — this avoids cross-attention interference between multiple images.

### Phase 3: Training & Validation Pipeline *(Weeks 5–8)*

| Task | Status | Deliverable |
|------|--------|-------------|
| Implement BLEU-1 and F1 metrics | ⬜ | `test_metrics.py` passes |
| Run `03_overfit_single_batch.py` | ⬜ | Loss → 0 on 5 examples |
| Fine-tune on VQA-RAD training set | ⬜ | Training curves in `results/` |
| Evaluate on VQA-RAD test set | ⬜ | Comparison table vs. Table 2 |

**Paper Baselines to Beat (VQA-RAD):**

| Model | BLEU-1 | F1 |
|-------|--------|----|
| Prior SOTA (specialist) | 71.03% | N/A |
| PaLM-E 84B (no fine-tuning) | 59.19% | 38.67% |
| Med-PaLM M 12B | 64.02% | 50.66% |
| Med-PaLM M 84B | 69.38% | 59.90% |
| **Med-PaLM M 562B** | **71.27%** | **62.06%** |

### Phase 4: Extended Experiments *(Month 3)*

| Task | Status | Deliverable |
|------|--------|-------------|
| Fine-tune on Slake-VQA | ⬜ | Cross-dataset comparison |
| Test one-shot exemplar ablation | ⬜ | With vs. without exemplar |
| Probe zero-shot generalization | ⬜ | Novel concept evaluation |
| Multi-task training (VQA-RAD + Slake) | ⬜ | Transfer learning analysis |

### Phase 5: Documentation & Portfolio *(Month 4)*

| Task | Status | Deliverable |
|------|--------|-------------|
| Write reproduction report | ⬜ | `docs/REPRODUCTION_LOG.md` |
| Create comparison tables | ⬜ | `results/tables/` |
| Generate qualitative examples | ⬜ | Model input → output screenshots |
| Final README polish | ⬜ | Portfolio-ready repository |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ~10GB disk space for datasets and model weights

### Installation

```bash
# Clone the repository
git clone https://github.com/Mrabbi3/biomed-multimodal-reproduction.git
cd biomed-multimodal-reproduction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Download

```bash
# VQA-RAD (small, publicly available — good starting point)
python data/download.py --dataset vqa_rad

# Slake-VQA
python data/download.py --dataset slake_vqa
```

### Run Sanity Check

```bash
# Verify data loading — displays sample image + question pair
python experiments/01_data_sanity_check.py
```

---

## Datasets Used

| Dataset | Task | Size | Access |
|---------|------|------|--------|
| **VQA-RAD** | Radiology VQA | 3,515 QA pairs, 315 images | Public |
| **Slake-VQA** | Radiology VQA (bilingual) | 14,028 QA pairs, 642 images | Public |
| **Path-VQA** | Pathology VQA | 32,799 QA pairs, 4,998 images | Public |
| MIMIC-CXR | CXR Report Generation | 377,110 images | Credentialed (PhysioNet) |

*This reproduction focuses on the first three VQA datasets for accessibility. MIMIC-CXR is an optional extension.*

---

## Evaluation Metrics

Following the paper's methodology:

- **BLEU-1**: Unigram precision between generated and reference answers
- **F1 (Token-level)**: Harmonic mean of token-level precision and recall
- **ROUGE-L**: Longest common subsequence overlap (for report generation)

> **Note from paper:** The authors use open-ended generative evaluation rather than classification accuracy, since their model generates free-form text. This is more challenging but better captures "near misses."

---

## Key Findings from the Paper

### What This Project Aims to Reproduce

1. **Generalist ≥ Specialist**: A single model with one set of weights can match task-specific models across multiple biomedical tasks (Table 2).

2. **Domain Fine-tuning Matters**: PaLM-E without biomedical fine-tuning scores 38.67% F1 on VQA-RAD vs. 62.06% after fine-tuning — a massive improvement from domain adaptation.

3. **Scaling Benefits Language Tasks Most**: Medical QA improves dramatically with scale (29% → 70% on MedQA), while image classification plateaus when the vision encoder isn't scaled.

4. **Positive Task Transfer**: Training on both CXR report generation AND classification simultaneously improves both tasks compared to training on either alone (Table 6).

5. **Zero-shot Generalization**: Med-PaLM M detects tuberculosis from chest X-rays at 87.68% accuracy despite never being trained on TB labels (Table 4).

---

## Differences from Original Paper

| Aspect | Med-PaLM M (Paper) | This Reproduction |
|--------|--------------------|--------------------|
| Base model | PaLM-E (proprietary) | LLaVA-Med / BLIP-2 (open-source) |
| Scale | 12B–562B params | ~7B–13B params |
| Training data | Full MultiMedBench (1M+) | VQA subset (~50K samples) |
| Compute | TPU pods | Single GPU (consumer) |
| Tasks | 14 simultaneous tasks | 2–3 VQA tasks |
| Evaluation | Automated + radiologist review | Automated metrics |

---

## References

```bibtex
@article{tu2023towards,
  title={Towards Generalist Biomedical AI},
  author={Tu, Tao and Azizi, Shekoofeh and Driess, Danny and others},
  journal={arXiv preprint arXiv:2307.14334},
  year={2023}
}
```

**Related Open-Source Projects:**
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) — Medical visual instruction tuning
- [BLIP-2](https://github.com/salesforce/LAVIS) — Bootstrapped language-image pretraining
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) — Open-source Flamingo reproduction

---

## Author

**MD Rabbi** — Computer Science Student & Aspiring AI/ML Engineer

*This project is part of an independent research initiative in biomedical multimodal AI, inspired by the Med-PaLM M paper from Google Research & Google DeepMind.*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

*Note: Datasets used in this project have their own licensing terms. Please review individual dataset licenses before use.*
