"""
Instruction prompt templates from Med-PaLM M paper.

These are extracted directly from Figure 2 and Table A.9-A.10 of the paper.
The one-shot exemplar technique uses a text-only example (image replaced
with <img> placeholder) to condition the model's output format.
"""

# ============================================================
# VQA Prompt (used for VQA-RAD, Slake-VQA, Path-VQA)
# Source: Table A.10
# ============================================================
VQA_INSTRUCTION = (
    "You are a helpful medical assistant. The following are questions "
    "about medical knowledge. Solve them in a step-by-step fashion, "
    "referring to authoritative sources as needed."
)

# ============================================================
# Chest X-ray Report Generation Prompt
# Source: Figure 2 (top)
# ============================================================
CXR_REPORT_INSTRUCTION = (
    "You are a helpful radiology assistant. "
    "Describe what lines, tubes, and devices are present and each of their locations. "
    "Describe if pneumothorax is present; if present, describe size on each side. "
    "Describe if pleural effusion is present; if present, describe amount on each side. "
    "Describe if lung opacity (atelectasis, fibrosis, consolidation, infiltrate, "
    "lung mass, pneumonia, pulmonary edema) is present; if present, describe kinds "
    "and locations. Describe the cardiac silhouette size. Describe the width and "
    "contours of the mediastinum. Describe if hilar enlargement is present; if "
    "enlarged, describe side. Describe what fractures or other skeletal "
    "abnormalities are present."
)

# ============================================================
# Dermatology Classification Prompt
# Source: Figure 2 (bottom)
# ============================================================
DERM_INSTRUCTION = (
    "You are a helpful dermatology assistant. The following are questions "
    "about skin lesions. Categorize the skin lesions into the most likely "
    "class given the patient history."
)

# ============================================================
# Radiology Report Summarization Prompt
# Source: Table A.11
# ============================================================
REPORT_SUMMARIZATION_INSTRUCTION = (
    "You are a helpful radiology assistant. The following are questions "
    "about radiology reports. Summarize the findings in the report into "
    "diagnostic statements."
)

# ============================================================
# CXR Classification Prompt
# Source: Table A.9
# ============================================================
CXR_CLASSIFICATION_INSTRUCTION = (
    "You are a helpful radiology assistant. The following are questions "
    "about findings in chest X-ray in different views. Identify if a "
    "specific type of abnormality is shown in the X-ray."
)

# ============================================================
# Mammography Classification Prompt
# Source: Table A.9
# ============================================================
MAMMO_INSTRUCTION = (
    "You are a helpful medical assistant. The following are questions "
    "about mammography reading. Provide a breast-level assessment based "
    "on the BI-RADS categories."
)

# ============================================================
# Genomic Variant Calling Prompt
# Source: Table A.9
# ============================================================
GENOMICS_INSTRUCTION = (
    "You are a helpful genetic assistant. The following are questions "
    "about variant calling. Identify the number of copies of the putative "
    "variant in pileup images."
)

# ============================================================
# Medical QA Prompt (MedQA, MedMCQA)
# Source: Table A.12, A.13
# ============================================================
MEDICAL_QA_INSTRUCTION = (
    "The following are multiple choice questions about medical knowledge. "
    "Solve them in a step-by-step fashion, starting by summarizing the "
    "available information. Output a single option from the four options "
    "as the final answer."
)

# ============================================================
# TB Zero-Shot Classification (for generalization experiments)
# Source: Figure 3
# ============================================================
TB_ZERO_SHOT_INSTRUCTION = (
    "You are a helpful radiology assistant. The following are questions "
    "about tuberculosis vs normal chest X-rays. Solve it step by step, "
    "output a Yes/No answer and explanation."
)


def build_vqa_prompt(
    question: str,
    exemplar_q: str = None,
    exemplar_a: str = None,
) -> str:
    """
    Build a VQA prompt following paper's one-shot exemplar format.

    Args:
        question: The question to answer
        exemplar_q: Optional example question (for one-shot)
        exemplar_a: Optional example answer (for one-shot)

    Returns:
        Formatted prompt string
    """
    parts = [f"Instructions: {VQA_INSTRUCTION}"]

    if exemplar_q and exemplar_a:
        parts.append(f"Given <img>. Q: {exemplar_q}")
        parts.append(f"A: {exemplar_a}")

    parts.append(f"Given <img>. Q: {question}")
    parts.append("A:")

    return "\n".join(parts)
