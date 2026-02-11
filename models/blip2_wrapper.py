"""
BLIP-2 Model Wrapper for Medical VQA.

BLIP-2 (Bootstrapped Language-Image Pre-training) is used as our open-source
alternative to Med-PaLM M's PaLM-E architecture. It combines a ViT vision
encoder with a language model via a Q-Former bridge.

This wrapper provides the same interface as BaseBiomedModel so we can
swap models easily.
"""

import torch
from PIL import Image
from typing import Optional

from .base_model import BaseBiomedModel


class BLIP2Wrapper(BaseBiomedModel):
    """
    BLIP-2 wrapper for medical visual question answering.

    Uses Salesforce's BLIP-2 with Flan-T5 as the language backbone.
    Smaller and more accessible than PaLM-E for reproduction work.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        device: str = None,
        load_in_8bit: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model ID
            device: 'cuda', 'cpu', or None for auto-detect
            load_in_8bit: Use 8-bit quantization (saves ~50% VRAM)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name, device)
        self.load_in_8bit = load_in_8bit

    def load_model(self) -> None:
        """Load BLIP-2 model and processor from HuggingFace."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"Loading BLIP-2: {self.model_name}")
        print(f"Device: {self.device} | 8-bit: {self.load_in_8bit}")

        self.processor = Blip2Processor.from_pretrained(self.model_name)

        load_kwargs = {}
        if self.load_in_8bit and self.device == "cuda":
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )

        if not self.load_in_8bit and self.device == "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"✓ Model loaded ({param_count:.1f}B parameters)")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        num_beams: int = 5,
    ) -> str:
        """
        Generate text response given an image and prompt.

        Args:
            image: PIL Image
            prompt: Text prompt / question
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            num_beams: Beam search width

        Returns:
            Generated text string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        # Move to device
        if not self.load_in_8bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
            )

        # Decode, skipping special tokens
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text.strip()

    def generate_batch(
        self,
        images: list,
        prompts: list,
        max_new_tokens: int = 256,
    ) -> list:
        """
        Generate responses for a batch of image-prompt pairs.

        Args:
            images: List of PIL Images
            prompts: List of text prompts
            max_new_tokens: Max tokens per response

        Returns:
            List of generated text strings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # BLIP-2 processor can handle batches
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        inputs = self.processor(
            images=rgb_images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        if not self.load_in_8bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in texts]
