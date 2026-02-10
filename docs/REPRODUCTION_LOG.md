# Reproduction Log

This document tracks progress, decisions, and findings throughout the reproduction study.

---

## Phase 1: Project Definition & Data Setup

**Start Date:** ___________

### Decisions Made

- **Target task(s):** Medical Visual Question Answering (VQA-RAD, Slake-VQA)
- **Rationale:** Publicly available, manageable size, clear evaluation metrics (BLEU-1, F1)

### Data Loading Results

| Dataset | Samples Loaded | Images OK | Questions OK |
|---------|---------------|-----------|--------------|
| VQA-RAD | _____ | ☐ | ☐ |
| Slake-VQA | _____ | ☐ | ☐ |

### Notes

_Record any issues, observations, or surprises here._

---

## Phase 2: Model Selection & Baseline

**Start Date:** ___________

### Model Choice

- **Selected model:** ___________
- **Rationale:** ___________
- **Parameters:** ___________

### Baseline Results (No Fine-tuning)

| Dataset | BLEU-1 | F1 | Notes |
|---------|--------|----| ------|
| VQA-RAD | _____ | _____ | |

---

## Phase 3: Training & Validation

**Start Date:** ___________

### Overfit Test (5 examples, 100 epochs)

- **Final loss:** ___________
- **Memorization achieved:** ☐ Yes ☐ No
- **Bug found:** ___________

### Full Training Results

| Model | Dataset | BLEU-1 | F1 | Epochs | Notes |
|-------|---------|--------|----| -------|-------|
| | VQA-RAD | _____ | _____ | | |

### Comparison to Paper (Table 2)

| Model | BLEU-1 | F1 |
|-------|--------|-----|
| Prior SOTA | 71.03% | N/A |
| Med-PaLM M 12B | 64.02% | 50.66% |
| Med-PaLM M 562B | 71.27% | 62.06% |
| **Ours** | _____ | _____ |

---

## Phase 4: Extended Experiments

_To be filled in during Phase 4._

---

## Phase 5: Final Analysis

_To be filled in during Phase 5._
