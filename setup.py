from setuptools import setup, find_packages

setup(
    name="biomed-multimodal-reproduction",
    version="0.1.0",
    author="MD Rabbi",
    description="Reproducing Med-PaLM M: Towards Generalist Biomedical AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "train": ["peft>=0.7.0", "accelerate>=0.25.0", "bitsandbytes"],
        "eval": ["nltk>=3.8.0", "rouge-score>=0.1.2"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0", "jupyter"],
        "dev": ["pytest>=7.4.0"],
    },
)
