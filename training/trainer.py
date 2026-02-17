"""
Training pipeline for medical VQA fine-tuning.

Implements the training loop following Med-PaLM M methodology:
- Instruction task prompting with one-shot exemplars
- Adafactor optimizer with momentum β1=0.9
- End-to-end fine-tuning on MultiMedBench tasks

For consumer GPUs, uses LoRA (Low-Rank Adaptation) instead of
full fine-tuning to fit within memory constraints.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from training.prompts import VQA_INSTRUCTION, build_vqa_prompt


class VQATrainingDataset(Dataset):
    """
    PyTorch Dataset wrapper for VQA training.

    Converts raw QA pairs into model-ready format with
    instruction prompts following the paper's approach.
    """

    def __init__(self, samples: list, processor, use_exemplar: bool = True):
        """
        Args:
            samples: List of dicts with 'image', 'question', 'answer'
            processor: Model processor (BLIP-2 or LLaVA)
            use_exemplar: Whether to prepend a one-shot exemplar
        """
        self.samples = samples
        self.processor = processor
        self.use_exemplar = use_exemplar

        # Fixed exemplar for one-shot prompting (paper Section 4.2)
        self.exemplar_q = "Is this a normal chest x-ray?"
        self.exemplar_a = "No."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build prompt following paper's format
        if self.use_exemplar:
            prompt = build_vqa_prompt(
                question=sample["question"],
                exemplar_q=self.exemplar_q,
                exemplar_a=self.exemplar_a,
            )
        else:
            prompt = build_vqa_prompt(question=sample["question"])

        image = sample["image"]
        if image is not None and image.mode != "RGB":
            image = image.convert("RGB")

        return {
            "image": image,
            "prompt": prompt,
            "answer": sample["answer"],
            "question": sample["question"],
        }


class MedVQATrainer:
    """
    Trainer for medical VQA fine-tuning.

    Supports two modes:
    1. Full fine-tuning (requires large GPU, 24GB+)
    2. LoRA fine-tuning (works on consumer GPUs, 8GB+)
    """

    def __init__(
        self,
        model,
        processor,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        output_dir: str = "results",
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        num_epochs: int = 10,
        use_lora: bool = True,
        lora_rank: int = 16,
        gradient_accumulation_steps: int = 4,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.device = next(model.parameters()).device
        self.training_log = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    def setup_lora(self):
        """Apply LoRA adapters for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # Auto-detect correct target module names for this model
            model_modules = [name for name, _ in self.model.named_modules()]
            
            # Try common naming conventions
            if any("q_proj" in m for m in model_modules):
                target = ["q_proj", "v_proj"]
            elif any(".q." in m or ".q" == m.split(".")[-1] for m in model_modules):
                target = ["q", "v"]
            else:
                # Fallback: find all linear layers in attention
                target = ["q", "v"]
            
            # Detect task type: encoder-decoder (T5) vs decoder-only
            if hasattr(self.model, "encoder") or "t5" in str(type(self.model)).lower():
                task = TaskType.SEQ_2_SEQ_LM
            else:
                task = TaskType.CAUSAL_LM

            print(f"  LoRA targets: {target}, task: {task}")

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=32,
                target_modules=target,
                lora_dropout=0.1,
                task_type=task,
            )
            self.model = get_peft_model(self.model, lora_config)

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"✓ LoRA applied: {trainable/1e6:.1f}M / {total/1e6:.1f}M params trainable "
                  f"({100*trainable/total:.2f}%)")

        except ImportError:
            print("Warning: peft not installed. Running full fine-tuning.")
            print("Install with: pip install peft")
            self.use_lora = False

    def collate_fn(self, batch):
        """Custom collate for variable-size inputs."""
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Process with model's processor
        # For training, we need input_ids for the full prompt+answer
        full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

        encoding = self.processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Create labels (mask the prompt portion)
        prompt_encodings = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        labels = encoding["input_ids"].clone()
        # Mask prompt tokens with -100 so loss only applies to answer
        for i in range(len(batch)):
            prompt_len = prompt_encodings["input_ids"][i].ne(
                self.processor.tokenizer.pad_token_id
            ).sum()
            labels[i, :prompt_len] = -100

        encoding["labels"] = labels
        return encoding

    def train(self):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  LoRA: {self.use_lora}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"{'='*60}\n")

        if self.use_lora:
            self.setup_lora()

        # Setup optimizer (paper uses Adafactor with β1=0.9)
        try:
            from transformers import Adafactor
            optimizer = Adafactor(
                self.model.parameters(),
                lr=self.learning_rate,
                relative_step=False,
                warmup_init=False,
            )
            print("Using Adafactor optimizer (paper's choice)")
        except ImportError:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
            )
            print("Using AdamW optimizer (Adafactor unavailable)")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,  # Safer for PIL images
        )

        self.model.train()
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for step, batch in enumerate(pbar):
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += outputs.loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

            avg_loss = epoch_loss / max(num_batches, 1)

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "global_step": global_step,
            }

            # Validation
            if self.val_dataset:
                val_loss = self._validate()
                log_entry["val_loss"] = val_loss
                print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best")
            else:
                print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

            self.training_log.append(log_entry)

        # Save final checkpoint and training log
        self._save_checkpoint("final")
        self._save_training_log()
        print(f"\n✓ Training complete. Results saved to {self.output_dir}/")

    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, f"checkpoint_{name}")
        os.makedirs(path, exist_ok=True)

        if self.use_lora:
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

        print(f"  Saved checkpoint: {path}")

    def _save_training_log(self):
        """Save training metrics to JSON."""
        log_path = os.path.join(self.output_dir, "logs", "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        print(f"  Saved training log: {log_path}")
