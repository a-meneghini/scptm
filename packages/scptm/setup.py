"""
setup.py
--------
Package metadata for SCPTM.
Install with:
    pip install -e .                   # editable dev install
    pip install -e ".[benchmark]"      # + benchmark deps (scikit-learn)
    pip install -e ".[udpipe]"         # + UDPipe lemmatisation backend
    pip install -e ".[keywords]"       # + RAKE keyphrase extraction
    pip install -e ".[full]"           # all optional deps
"""

import re
from pathlib import Path

from setuptools import find_packages, setup


def _read_version() -> str:
    """Single source of truth: scptm/__init__.py.

    Parsed with a regex instead of importing the package, because importing it
    at build time would pull in torch / sentence-transformers.
    """
    init_py = Path(__file__).parent / "src" / "scptm" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        init_py.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Unable to find __version__ in {init_py}")
    return match.group(1)


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scptm",
    version=_read_version(),
    author="Alessandro Meneghini",
    author_email="alessandro.meneghini@uniud.it",
    description=(
        "SCPTM: Structural Contextual Probabilistic Topic Model — "
        "a VAE-GNN topic model with syntactic dependency graphs, "
        "contextual word embeddings, and beta temperature scaling."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a-meneghini/scptm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["scptm=scptm.cli:main"]},
    python_requires=">=3.10",
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
            "contextualized-topic-models>=2.3",
            "vaderSentiment>=3.3",
            "gensim>=4.3",
        ],
        "full": [
            "bertopic>=0.15",
            "contextualized-topic-models>=2.3",
            "vaderSentiment>=3.3",
            "gensim>=4.3",
            "pacmap>=0.7",
            "ufal.udpipe>=1.3",
            "rake-nltk>=1.0",
        ],
        "udpipe": [
            "ufal.udpipe>=1.3",
        ],
        "keywords": [
            "rake-nltk>=1.0",
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
