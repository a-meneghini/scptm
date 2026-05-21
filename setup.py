"""
setup.py
--------
Package metadata for SCPTM.
Install with:
    pip install -e .                   # editable dev install
    pip install -e ".[benchmark]"      # + benchmark deps (scikit-learn)
    pip install -e ".[full]"           # all optional deps
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scptm",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description=(
        "SCPTM: Structural Contextual Probabilistic Topic Model — "
        "a VAE-GNN topic model with syntactic dependency graphs and "
        "contextual word embeddings."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/scptm",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torch-geometric>=2.4",
        "sentence-transformers>=2.2",
        "spacy>=3.5",
        "scikit-learn>=1.2",
        "numpy>=1.24",
        "pandas>=1.5",
        "scipy>=1.10",
        "umap-learn>=0.5",
        "matplotlib>=3.7",
        "plotly>=5.14",
        "tqdm>=4.65",
    ],
    extras_require={
        "benchmark": [
            "bertopic>=0.15",
            "tritopic>=2.3",
        ],
        "full": [
            "bertopic>=0.15",
            "tritopic>=2.3",
            "pacmap>=0.7",
        ],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "ruff>=0.1",
            "mypy>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    keywords=[
        "topic-modeling", "nlp", "graph-neural-network",
        "variational-autoencoder", "sentence-transformers",
        "dependency-parsing", "bertopic-alternative",
    ],
)
