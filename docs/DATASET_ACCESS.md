# Dataset Access Guide

## Publicly Available Datasets (No Approval Needed)

| Dataset | Access | Download Command |
|---------|--------|-----------------|
| **VQA-RAD** | ✅ Open | `python data/download.py --dataset vqa_rad` |
| **Slake-VQA** | ✅ Open | `python data/download.py --dataset slake_vqa` |
| **Path-VQA** | ✅ Open | `python data/download.py --dataset path_vqa` |

## Credentialed Access Required

| Dataset | Access | How to Get Access |
|---------|--------|-------------------|
| **MIMIC-CXR** | 🔒 PhysioNet | 1. Create account at [physionet.org](https://physionet.org) |
|  |  | 2. Complete CITI training |
|  |  | 3. Sign data use agreement |
|  |  | 4. Request access (takes 1-2 weeks) |
| **MIMIC-III** | 🔒 PhysioNet | Same process as MIMIC-CXR |

## Recommended Starting Order

1. **VQA-RAD** — smallest dataset, fastest to experiment with
2. **Slake-VQA** — more data, useful for cross-dataset evaluation
3. **Path-VQA** — different modality (pathology), good for generalization tests
4. **MIMIC-CXR** — only if pursuing report generation experiments
