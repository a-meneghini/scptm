"""
scptm/nlp.py
------------
NLP pipeline initialisation: SBERT + spaCy.
Supports English and Italian.
"""

import torch
from sentence_transformers import SentenceTransformer


def setup_nlp_pipeline(lang: str = "eng", verbose: bool = True):
    """
    Load SBERT (on GPU if available) and spaCy (CPU).

    Parameters
    ----------
    lang : str
        "eng" (default) or "ita".
    verbose : bool
        Print device info.

    Returns
    -------
    sbert_model : SentenceTransformer
    nlp : spacy Language
    stop_words : list | str
        Stop word list for CountVectorizer.
    """
    import spacy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if lang == "ita":
        sbert_name = "paraphrase-multilingual-MiniLM-L12-v2"
        spacy_model = "it_core_news_sm"
    else:
        sbert_name = "all-MiniLM-L6-v2"
        spacy_model = "en_core_web_sm"

    sbert = SentenceTransformer(sbert_name, device=device)

    try:
        nlp = spacy.load(spacy_model, disable=["ner"])
    except OSError:
        from spacy.cli import download
        download(spacy_model)
        nlp = spacy.load(spacy_model, disable=["ner"])

    nlp.max_length = 2_000_000
    stop_words = list(nlp.Defaults.stop_words) if lang == "ita" else "english"

    if verbose:
        print(f"NLP pipeline: SBERT='{sbert_name}' on {sbert.device}, spaCy='{spacy_model}' on CPU")

    return sbert, nlp, stop_words
