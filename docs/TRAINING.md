# Training Guide

Complete guide to reproducing our results. This covers environment setup, training, evaluation, and troubleshooting.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | T4 16GB (Colab free) | A100 40GB |
| RAM | 12GB | 32GB |
| Storage | 25GB | 50GB |
| Python | 3.10+ | 3.11 |
| Training time | ~35 min (T4) | ~10 min (A100) |

## Step 1: Environment Setup

### Google Colab (Recommended)

```python
!git clone https://github.com/Mrabbi3/biomed-multimodal-reproduction.git
%cd biomed-multimodal-reproduction
!pip install -q transformers>=4.36.0 accelerate>=0.25.0 peft>=0.7.0 \
    bitsandbytes datasets Pillow tqdm pyyaml nltk rouge-score matplotlib seaborn evaluate
```

### Local

```bash
git clone https://github.com/Mrabbi3/biomed-multimodal-reproduction.git
cd biomed-multimodal-reproduction
pip install -r requirements.txt
```

## Step 2: Download Data

```bash
python data/download.py --dataset vqa_rad
```

This downloads VQA-RAD from HuggingFace (3,515 QA pairs, 315 radiology images). Takes ~1 minute.

## Step 3: BLIP-2 Wrapper Patch (Colab Only)

If using transformers >= 4.36, the BLIP-2 wrapper needs `BitsAndBytesConfig` instead of the deprecated `load_in_8bit` kwarg. The repo already has this fix, but if you see errors about `load_in_8bit`, apply this patch:

```python
# In models/blip2_wrapper.py, replace:
load_kwargs["load_in_8bit"] = True

# With:
from transformers import BitsAndBytesConfig
load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
```

## Step 4: Training

### Option A: Interactive Training (Recommended for Colab)

This runs training directly in your Python session — avoids subprocess crashes:

```python
import sys, torch
sys.path.insert(0, '.')
from models.blip2_wrapper import BLIP2Wrapper
from data.vqa_rad_loader import VQARadDataset
from peft import LoraConfig, get_peft_model, TaskType

# Load model
model = BLIP2Wrapper(load_in_8bit=True)
model.load_model()

# Load data
train_ds = VQARadDataset(data_dir="data/vqa_rad", split="train")

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q", "v"],  # Flan-T5 attention layer names
    lora_dropout=0.1,
    task_type=TaskType.SEQ_2_SEQ_LM,  # T5 is encoder-decoder
)
model.model.train()
peft_model = get_peft_model(model.model, lora_config)

# Train
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, peft_model.parameters()), lr=5e-5
)

for epoch in range(5):
    epoch_loss, n = 0, 0
    for i in range(min(len(train_ds), 500)):
        s = train_ds[i]
        img = s.get("image_processed") or s.get("image")
        if img.mode != "RGB": img = img.convert("RGB")
        
        inputs = model.processor(
            images=img,
            text=f"Question: {s['question']} Answer:",
            return_tensors="pt"
        )
        inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}
        labels = model.processor.tokenizer(
            s["answer"], return_tensors="pt", truncation=True
        )
        label_ids = labels["input_ids"].to(peft_model.device)
        
        try:
            out = peft_model(**inputs, labels=label_ids)
            if out.loss is not None:
                out.loss.backward()
                if (n+1) % 8 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_loss += out.loss.item()
                n += 1
        except: continue
    
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss/max(n,1):.4f}")
```

### Option B: Script-Based Training

```bash
python experiments/04_train_vqa.py \
    --dataset vqa_rad \
    --epochs 10 \
    --batch_size 1 \
    --lr 5e-5 \
    --lora_rank 8 \
    --grad_accum 8 \
    --quantize \
    --use_exemplar
```

**Note:** Script-based training may crash silently on T4 GPUs due to memory constraints. If this happens, use Option A instead.

## Step 5: Evaluation

```python
from evaluation.metrics import compute_bleu1, compute_f1_token

peft_model.eval()
test_ds = VQARadDataset(data_dir="data/vqa_rad", split="test")
preds, truths = [], []

for i in range(len(test_ds)):
    s = test_ds[i]
    img = s.get("image_processed") or s.get("image")
    if img.mode != "RGB": img = img.convert("RGB")
    inputs = model.processor(images=img, text=f"Question: {s['question']} Answer:", return_tensors="pt")
    inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = peft_model.generate(**inputs, max_new_tokens=32)
    preds.append(model.processor.batch_decode(out, skip_special_tokens=True)[0].strip())
    truths.append(s["answer"])

bleu = sum(compute_bleu1(p,t) for p,t in zip(preds,truths)) / len(preds) * 100
f1 = sum(compute_f1_token(p,t) for p,t in zip(preds,truths)) / len(preds) * 100
print(f"BLEU-1: {bleu:.2f}% | F1: {f1:.2f}%")
```

## Step 6: Generate Report

```bash
python evaluation/compare_to_paper.py
```

This creates comparison charts in `results/figures/` and markdown tables in `results/tables/`.

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Salesforce/blip2-flan-t5-xl | 3.4B parameters |
| Quantization | 8-bit (bitsandbytes) | ~10GB VRAM |
| LoRA rank | 8 | 4.7M trainable params (0.12%) |
| LoRA alpha | 16 | |
| LoRA targets | q, v | Flan-T5 attention layers |
| Task type | SEQ_2_SEQ_LM | T5 is encoder-decoder |
| Learning rate | 5e-5 | |
| Optimizer | AdamW | |
| Gradient accumulation | 8 | Effective batch size = 8 |
| Epochs | 5 | |
| Training samples | 500 (of 1,793) | |

## Troubleshooting

### `load_in_8bit` error
Newer transformers versions require `BitsAndBytesConfig`. See Step 3 above.

### `Target modules {'v_proj', 'q_proj'} not found`
BLIP-2's Flan-T5 uses `q` and `v`, not `q_proj` and `v_proj`. Fix in `training/trainer.py`:
```python
target_modules=["q", "v"]  # NOT ["q_proj", "v_proj"]
```

### `TaskType.CAUSAL_LM` error
Flan-T5 is encoder-decoder, not causal. Use:
```python
task_type=TaskType.SEQ_2_SEQ_LM  # NOT CAUSAL_LM
```

### Training crashes silently on Colab
The T4 GPU has limited memory. Solutions:
- Use interactive training (Option A) instead of subprocess
- Reduce `lora_rank` to 4
- Use `batch_size=1` with higher `grad_accum`

### Model outputs all "yes"
The model collapsed into a single response. Train for more epochs on more data, or reduce learning rate to 3e-5.

## Expected Results

| Stage | BLEU-1 (%) | F1 (%) |
|-------|-----------|--------|
| Zero-shot baseline | 0.44 | 0.70 |
| + One-shot exemplar | 2.54 | 3.86 |
| + Fine-tuning (5ep) | 26.16 | 26.16 |
