"""
LLaVA-Med Model Wrapper for Medical VQA.

LLaVA-Med is a medical adaptation of LLaVA (Large Language and Vision Assistant),
fine-tuned on biomedical image-text data from PubMed. It's the closest open-source
equivalent to Med-PaLM M's approach of domain-specific fine-tuning.

Reference: Li et al., "LLaVA-Med: Training a Large Language-and-Vision Assistant
for Biomedicine in One Day" (2023)
"""

import torch
from PIL import Image
from typing import Optional

from .base_model import BaseBiomedModel


class LLaVAMedWrapper(BaseBiomedModel):
    """
    LLaVA-Med wrapper for medical visual question answering.

    Note: LLaVA-Med requires the llava package to be installed separately.
    If unavailable, falls back to standard LLaVA or raises instructions.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llava-med-v1.5-mistral-7b",
        device: str = None,
        load_in_4bit: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name, device)
        self.load_in_4bit = load_in_4bit

    def load_model(self) -> None:
        """
        Load LLaVA-Med model.

        Attempts to load via transformers' LlavaForConditionalGeneration.
        Falls back with helpful error if specific dependencies are missing.
        """
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor

            print(f"Loading LLaVA-Med: {self.model_name}")
            print(f"Device: {self.device} | 4-bit: {self.load_in_4bit}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)

            load_kwargs = {"torch_dtype": torch.float16}
            if self.load_in_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"

            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )

            if not self.load_in_4bit and self.device == "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"✓ Model loaded ({param_count:.1f}B parameters)")

        except ImportError as e:
            print(f"Error loading LLaVA-Med: {e}")
            print("Try: pip install transformers>=4.36.0 accelerate bitsandbytes")
            raise
        except OSError as e:
            print(f"Model not found: {e}")
            print(f"The model '{self.model_name}' may require authentication or may not exist.")
            print("Falling back to BLIP-2 is recommended.")
            raise

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response given image and prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # LLaVA expects conversation format
        conversation_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            images=image,
            text=conversation_prompt,
            return_tensors="pt",
        )

        if not self.load_in_4bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode only the new tokens (skip the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )[0]

        return generated_text.strip()
