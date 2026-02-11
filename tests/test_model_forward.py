"""
Unit tests for model wrappers.

Tests the model interface without requiring actual model weights.
For tests with real models, use the experiment scripts.

Run: pytest tests/test_model_forward.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseBiomedModel
from training.prompts import (
    VQA_INSTRUCTION, CXR_REPORT_INSTRUCTION, DERM_INSTRUCTION,
    build_vqa_prompt,
)


class TestPromptFormatting:
    """Test that prompts match the paper's format (Figure 2)."""

    def test_vqa_prompt_basic(self):
        prompt = build_vqa_prompt(question="Is this normal?")
        assert "Instructions:" in prompt
        assert "Q: Is this normal?" in prompt
        assert prompt.endswith("A:")

    def test_vqa_prompt_with_exemplar(self):
        prompt = build_vqa_prompt(
            question="What organ is shown?",
            exemplar_q="Is the lung healthy?",
            exemplar_a="No.",
        )
        assert "Is the lung healthy?" in prompt
        assert "A: No." in prompt
        assert "What organ is shown?" in prompt
        # Exemplar should come before the actual question
        assert prompt.index("Is the lung healthy?") < prompt.index("What organ is shown?")

    def test_vqa_prompt_uses_img_placeholder(self):
        """Paper: one-shot exemplar uses <img> placeholder."""
        prompt = build_vqa_prompt(
            question="test",
            exemplar_q="example",
            exemplar_a="answer",
        )
        assert "<img>" in prompt

    def test_all_instructions_nonempty(self):
        """Verify all instruction templates from the paper are populated."""
        from training.prompts import (
            VQA_INSTRUCTION, CXR_REPORT_INSTRUCTION, DERM_INSTRUCTION,
            REPORT_SUMMARIZATION_INSTRUCTION, CXR_CLASSIFICATION_INSTRUCTION,
            MAMMO_INSTRUCTION, GENOMICS_INSTRUCTION, MEDICAL_QA_INSTRUCTION,
            TB_ZERO_SHOT_INSTRUCTION,
        )
        instructions = [
            VQA_INSTRUCTION, CXR_REPORT_INSTRUCTION, DERM_INSTRUCTION,
            REPORT_SUMMARIZATION_INSTRUCTION, CXR_CLASSIFICATION_INSTRUCTION,
            MAMMO_INSTRUCTION, GENOMICS_INSTRUCTION, MEDICAL_QA_INSTRUCTION,
            TB_ZERO_SHOT_INSTRUCTION,
        ]
        for inst in instructions:
            assert len(inst) > 20, f"Instruction too short: {inst[:50]}"


class TestBaseModel:
    """Test the abstract base model interface."""

    def test_format_prompt(self):
        """Test the base class prompt formatter."""
        # Create a concrete subclass for testing
        class DummyModel(BaseBiomedModel):
            def load_model(self): pass
            def generate(self, image, prompt, max_new_tokens=256): return "test"

        model = DummyModel("test_model", "cpu")
        prompt = model.format_prompt(
            question="What is shown?",
            instruction="You are a helpful assistant.",
            exemplar_q="Is this normal?",
            exemplar_a="Yes.",
        )

        assert "Instructions:" in prompt
        assert "What is shown?" in prompt
        assert "Is this normal?" in prompt
        assert "A: Yes." in prompt


class TestMultiTaskMixer:
    """Test the multi-task data mixer."""

    def test_mixer_creation(self):
        pytest = __import__("pytest")
        try:
            from torch.utils.data import Dataset
        except ImportError:
            pytest.skip("torch not installed")
            return

        from training.multitask_mixer import MultiTaskMixer

        class FakeDataset(Dataset):
            def __init__(self, n):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return {"question": f"q{idx}", "answer": f"a{idx}"}

        mixer = MultiTaskMixer({
            "dataset_a": FakeDataset(100),
            "dataset_b": FakeDataset(50),
        })

        assert len(mixer) == 150  # Sum of both
        sample = mixer[0]
        assert "dataset_name" in sample

    def test_mixer_custom_ratios(self):
        pytest = __import__("pytest")
        try:
            from torch.utils.data import Dataset
        except ImportError:
            pytest.skip("torch not installed")
            return
        from training.multitask_mixer import MultiTaskMixer

        class FakeDataset(Dataset):
            def __init__(self, n):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return {"question": f"q{idx}", "answer": f"a{idx}"}

        mixer = MultiTaskMixer(
            {"a": FakeDataset(100), "b": FakeDataset(100)},
            mixture_ratios={"a": 0.8, "b": 0.2},
            total_samples_per_epoch=100,
        )

        assert len(mixer) == 100
