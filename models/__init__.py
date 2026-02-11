"""
Model wrappers for multimodal biomedical AI.
Provides unified interface for different backbone models.

Available models:
    - BLIP2Wrapper: BLIP-2 with Flan-T5 (recommended, easiest to run)
    - LLaVAMedWrapper: LLaVA-Med (closest to Med-PaLM M approach)
"""

from .base_model import BaseBiomedModel

__all__ = ["BaseBiomedModel"]
