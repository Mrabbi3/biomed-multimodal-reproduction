# Reproducing Med-PaLM M: An Open-Source Approach to Generalist Biomedical AI

[![Paper](https://img.shields.io/badge/Paper-arXiv%202307.14334-red)](https://arxiv.org/abs/2307.14334)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mrabbi3/biomed-multimodal-reproduction/blob/main/notebooks/Full_Reproduction_Colab.ipynb)

An independent reproduction study of [Med-PaLM M](https://arxiv.org/abs/2307.14334) (Tu et al., 2023) — the first generalist biomedical AI — using exclusively open-source models. We replace Google's proprietary PaLM-E (562B) with BLIP-2 (3.4B), implement the paper's complete medical VQA pipeline, and achieve a **59x improvement** over zero-shot baseline through LoRA fine-tuning on a single free GPU.

**Author:** MD Rabbi · Department of Computer Science · February 2026

---

## Results

### Main Performance Comparison (Paper Table 2)

| Model | Parameters | BLEU-1 (%) | F1 (%) | Source |
|-------|-----------|-----------|--------|--------|
| Prior SOTA (specialist) | Various | 71.03 | — | Paper Table 2 |
| PaLM-E 84B (zero-shot) | 84B | 59.19 | 38.67 | Paper Table 2 |
| Med-PaLM M 12B | 12B | 64.02 | 50.66 | Paper Table 2 |
| Med-PaLM M 84B | 84B | 69.38 | 59.90 | Paper Table 2 |
| Med-PaLM M 562B | 562B | 71.27 | 62.06 | Paper Table 2 |
| **Ours: BLIP-2 (zero-shot)** | **3.4B** | **0.44** | **0.70** | **This work** |
| **Ours: BLIP-2 (fine-tuned)** | **3.4B** | **26.16** | **26.16** | **This work** |

### Progressive Improvement

| Stage | BLEU-1 (%) | Improvement |
|-------|-----------|-------------|
| Zero-shot baseline | 0.44 | — |
| + One-shot exemplar | 2.54 | 5.8x |
| + LoRA fine-tuning (5 epochs) | 26.16 | 59x |

### Key Findings

1. **Fine-tuning works dramatically** — LoRA fine-tuning on just 500 samples for 5 epochs improved BLEU-1 from 0.44% to 26.16%, a 59x improvement
2. **One-shot exemplars help** — The paper's prompting strategy produced a 2.7x BLEU-1 improvement (0.95% → 2.54%), independently confirming this technique transfers across model scales
3. **Format learning is critical** — Zero-shot BLIP-2 generates verbose descriptions ("there is no evidence of an aortic aneurysm") instead of expected concise answers ("yes"). Fine-tuning teaches the correct output format
4. **Scale still matters** — Despite dramatic improvement, the 165x parameter gap (3.4B vs 562B) results in a significant absolute performance difference (26.16% vs 71.27%)

### Figures

<p align="center">
  <img src="results/figures/vqa_rad_comparison.png" width="700" alt="VQA-RAD Comparison"/>
  <br><em>Figure 1: Our fine-tuned BLIP-2 (3.4B) vs Med-PaLM M baselines on VQA-RAD</em>
</p>

<p align="center">
  <img src="results/figures/improvement_progression.png" width="600" alt="Improvement Progression"/>
  <br><em>Figure 2: Progressive improvement from zero-shot → exemplar → fine-tuned</em>
</p>

<p align="center">
  <img src="results/figures/training_curve.png" width="600" alt="Training Curve"/>
  <br><em>Figure 3: Training loss curve showing consistent learning over 5 epochs</em>
</p>

<p align="center">
  <img src="results/figures/exemplar_ablation.png" width="500" alt="Exemplar Ablation"/>
  <br><em>Figure 4: One-shot exemplar ablation — 2.7x improvement confirms paper's methodology</em>
</p>

---

## What This Project Does

Med-PaLM M showed that a single AI model can handle radiology VQA, pathology analysis, report generation, and more — all at once. But it uses PaLM-E, a proprietary 562-billion-parameter model nobody outside Google can access.

This project asks: **can we reproduce the methodology with open-source tools?** We implement the full pipeline — from data loading to evaluation — and test it on VQA-RAD (3,515 radiology question-answer pairs). We validate key methodological claims and provide a complete, runnable codebase for the research community.

---

## Quick Start

### Google Colab (Recommended)

Click the Colab badge above. Select **T4 GPU** runtime. Run the setup cell, then the training cell. Full pipeline takes ~1 hour.

### Local Installation

```bash
git clone https://github.com/Mrabbi3/biomed-multimodal-reproduction.git
cd biomed-multimodal-reproduction
pip install -r requirements.txt
python data/download.py --dataset vqa_rad
```

See [TRAINING.md](docs/TRAINING.md) for the complete training guide.

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
│  Instruction + Question ────┘    + LoRA adapters         │
└─────────────────────────────────────────────────────────┘
```

| Component | Med-PaLM M | Our Reproduction |
|-----------|-----------|------------------|
| Language model | PaLM-E (562B) | Flan-T5-XL (3B) via BLIP-2 |
| Vision-language bridge | Linear projection | Q-Former (32 learned queries) |
| Training | Full fine-tuning on TPU v4 | LoRA (rank=8) on single T4 GPU |
| Compute time | Weeks | ~1 hour |
| Cost | $100K+ (estimated) | Free (Google Colab) |

---

## Repository Structure

```
biomed-multimodal-reproduction/
├── models/                          # Model wrappers
│   ├── blip2_wrapper.py             #   BLIP-2 with 8-bit quantization
│   ├── llava_med_wrapper.py         #   LLaVA-Med (alternative)
│   └── base_model.py               #   Abstract interface
├── data/                            # Data pipeline
│   ├── download.py                  #   HuggingFace dataset download
│   ├── vqa_rad_loader.py            #   VQA-RAD (3,515 QA pairs)
│   ├── slake_loader.py              #   Slake-VQA (14,028 QA pairs)
│   └── preprocessing.py             #   224x224 resize, grayscale→RGB
├── training/                        # Training pipeline
│   ├── trainer.py                   #   LoRA fine-tuning with SEQ_2_SEQ_LM
│   ├── prompts.py                   #   9 instruction templates from paper
│   └── multitask_mixer.py           #   Proportional task sampling
├── evaluation/                      # Evaluation suite
│   ├── metrics.py                   #   BLEU-1, token F1
│   ├── evaluate.py                  #   Full test set evaluation
│   └── compare_to_paper.py          #   Table 2 comparison + charts
├── experiments/                     # 5-phase experiment pipeline
│   ├── 01_data_sanity_check.py
│   ├── 02_forward_pass_test.py      #   Zero-shot baseline
│   ├── 03_overfit_single_batch.py   #   Training validation
│   ├── 04_train_vqa.py              #   Full training
│   └── 05_zero_shot_eval.py         #   Exemplar ablation
├── results/                         # Generated outputs
│   ├── figures/                     #   PNG charts
│   ├── tables/                      #   JSON metrics + markdown comparison
│   └── logs/                        #   Training logs
├── notebooks/                       # Google Colab notebook
├── docs/                            # Guides
│   ├── TRAINING.md                  #   Complete training guide
│   ├── SETUP_GUIDE.md
│   └── DATASET_ACCESS.md
├── tests/                           # 27 unit tests
└── configs/                         # Hyperparameters from paper
```

---

## Datasets

| Dataset | QA Pairs | Images | Modalities | Used For |
|---------|----------|--------|------------|----------|
| [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | 3,515 | 315 | CT, MRI, X-ray | Primary evaluation |
| [Slake-VQA](https://huggingface.co/datasets/mdwiratathya/SLAKE) | 14,028 | 642 | CT, MRI, X-ray | Cross-dataset transfer |
| [Path-VQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | 32,799 | 4,998 | Pathology | Future work |

---

## Citation

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
