"""
Abstract base class for multimodal biomedical models.
All model wrappers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from PIL import Image


class BaseBiomedModel(ABC):
    """
    Abstract base class for biomedical multimodal models.

    Any model wrapper (LLaVA-Med, BLIP-2, etc.) must implement:
    - load_model(): Initialize model and tokenizer
    - generate(): Produce text from image + text input
    - format_prompt(): Build the instruction prompt (paper Section 4.2)
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load pretrained model weights and tokenizer."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate text response given an image and prompt.

        Args:
            image: PIL Image (already preprocessed to 224x224)
            prompt: Full instruction prompt with question
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        pass

    def format_prompt(
        self,
        question: str,
        instruction: str,
        exemplar_q: Optional[str] = None,
        exemplar_a: Optional[str] = None,
    ) -> str:
        """
        Format the instruction prompt following Med-PaLM M's approach.

        Paper (Section 4.2):
        "We provided the model with task-specific instructions to prompt
         the model to perform different types of tasks in a unified
         generative framework."

        The prompt structure:
        1. Task instruction
        2. One-shot exemplar (text-only, image replaced with <img>)
        3. Actual question with image

        Args:
            question: The question to answer
            instruction: Task-specific instruction text
            exemplar_q: Optional example question for one-shot
            exemplar_a: Optional example answer for one-shot

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Instructions: {instruction}"]

        # Add one-shot exemplar if provided (paper: text-only with <img> placeholder)
        if exemplar_q and exemplar_a:
            prompt_parts.append(f"Given <img>. Q: {exemplar_q}")
            prompt_parts.append(f"A: {exemplar_a}")

        # Add the actual question
        prompt_parts.append(f"Given <img>. Q: {question}")
        prompt_parts.append("A:")

        return "\n".join(prompt_parts)
