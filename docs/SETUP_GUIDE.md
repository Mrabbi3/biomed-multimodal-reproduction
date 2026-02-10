# Setup Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- Git

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Mrabbi3/biomed-multimodal-reproduction.git
cd biomed-multimodal-reproduction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Hugging Face token (for model downloads)
```

### 5. Download Datasets

```bash
# Start with VQA-RAD (smallest, ~50MB)
python data/download.py --dataset vqa_rad

# Optional: additional datasets
python data/download.py --dataset slake_vqa
python data/download.py --dataset path_vqa
```

### 6. Run Sanity Checks

```bash
# Verify everything works
python experiments/01_data_sanity_check.py

# Run unit tests
pytest tests/ -v
```

## GPU Setup Notes

For consumer GPUs (RTX 3060/4060 with 8-12GB VRAM):
- Use quantized models (4-bit or 8-bit)
- Batch size of 1-4
- Consider gradient checkpointing

For cloud GPUs (A100/V100):
- Full precision fine-tuning possible
- Batch sizes of 8-16
