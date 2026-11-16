"""
tests/test_nlp_backends.py
==========================
Tests for the pluggable lemmatisation backends (spaCy / UDPipe).

They are intentionally light: no model is downloaded and no network access is
required.  The end-to-end parse tests run only when a ``.udpipe`` model is
already available locally — through ``SCPTM_UDPIPE_MODEL_ENG`` /
``SCPTM_UDPIPE_MODEL_ITA`` or the default cache directory.

Run with pytest::

    pytest tests/test_nlp_backends.py -v

or directly (no pytest required)::

    python tests/test_nlp_backends.py
"""

import os
import tempfile
import traceback
from pathlib import Path

from scptm.config import SCPTMConfig
from scptm.nlp import (
    LEMMATIZERS,
    UDPipeDoc,
    UDPipeToken,
    _default_udpipe_cache_dir,
    _resolve_udpipe_model_path,
    _vectorizer_stop_words,
    get_stop_words,
)


def _restore_env(name: str, previous):
    """Restore an environment variable to its pre-test value (or unset it)."""
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_config_defaults_to_spacy():
    assert SCPTMConfig().lemmatizer == "spacy"
    assert SCPTMConfig().udpipe_model_path is None
    assert SCPTMConfig().udpipe_cache_dir is None


def test_config_accepts_supported_backends():
    for backend in LEMMATIZERS:
        cfg = SCPTMConfig(lang="ita", lemmatizer=backend)
        assert cfg.lemmatizer == backend


def test_config_rejects_unknown_backend():
    raised = False
    try:
        SCPTMConfig(lemmatizer="stanza")
    except ValueError:
        raised = True
    assert raised, "SCPTMConfig must reject an unsupported lemmatizer (ValueError)"


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

def test_stop_words_are_language_specific():
    assert "the" in get_stop_words("eng")
    assert "il" in get_stop_words("ita")
    assert len(get_stop_words("eng")) > 50
    assert len(get_stop_words("ita")) > 50


def test_vectorizer_stop_words_preserve_legacy_shapes():
    # English keeps sklearn's built-in "english" keyword; Italian gets a list.
    assert _vectorizer_stop_words("eng") == "english"
    ita = _vectorizer_stop_words("ita")
    assert isinstance(ita, list) and ita and all(isinstance(w, str) for w in ita)


# ---------------------------------------------------------------------------
# UDPipe token adapter (no model needed)
# ---------------------------------------------------------------------------

def test_udpipe_token_and_doc_interface():
    stop = frozenset({"the", "is"})
    head = UDPipeToken(1, "runs", "run", "VERB", "VBZ", "root", stop)
    dep = UDPipeToken(2, "dogs", "dog", "NOUN", "NNS", "nsubj", stop)
    dep.head = head
    doc = UDPipeDoc([head, dep])

    assert len(doc) == 2
    assert [t.text for t in doc] == ["runs", "dogs"]
    assert doc[1].head is head
    assert doc[1].head.lemma_ == "run"
    assert doc[1].dep_ == "nsubj"
    assert doc[1].is_alpha is True
    assert UDPipeToken(1, "the", "the", "DET", "DT", "det", stop).is_stop is True
    assert UDPipeToken(1, "42", "42", "NUM", "CD", "nummod", stop).is_alpha is False


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def test_missing_explicit_model_path_raises():
    with tempfile.TemporaryDirectory() as tmp:
        raised = False
        try:
            _resolve_udpipe_model_path(
                "eng", model_path=str(Path(tmp) / "does_not_exist.udpipe")
            )
        except FileNotFoundError:
            raised = True
        assert raised, "a missing explicit model path must raise FileNotFoundError"


def test_env_var_overrides_cache_dir():
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "custom.udpipe"
        model.write_bytes(b"not-a-real-model")
        previous = os.environ.get("SCPTM_UDPIPE_MODEL_ITA")
        os.environ["SCPTM_UDPIPE_MODEL_ITA"] = str(model)
        try:
            assert _resolve_udpipe_model_path("ita") == model
        finally:
            _restore_env("SCPTM_UDPIPE_MODEL_ITA", previous)


def test_default_cache_dir_honours_env():
    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("SCPTM_UDPIPE_MODEL_DIR")
        os.environ["SCPTM_UDPIPE_MODEL_DIR"] = tmp
        try:
            assert _default_udpipe_cache_dir() == Path(tmp)
        finally:
            _restore_env("SCPTM_UDPIPE_MODEL_DIR", previous)


# ---------------------------------------------------------------------------
# End-to-end parse (only when a model is already present locally)
# ---------------------------------------------------------------------------

_UDPIPE_FILENAMES = {
    "eng": "english-ewt-ud-2.5-191206.udpipe",
    "ita": "italian-isdt-ud-2.5-191206.udpipe",
}


def _local_model(lang: str):
    env = os.environ.get(f"SCPTM_UDPIPE_MODEL_{lang.upper()}")
    if env and Path(env).is_file():
        return env
    cached = _default_udpipe_cache_dir() / _UDPIPE_FILENAMES[lang]
    return str(cached) if cached.is_file() else None


def test_udpipe_parse_smoke():
    cases = [
        ("eng", "The dogs are running quickly in the park.", "dog"),
        ("ita", "I cani stanno correndo velocemente nel parco.", "cane"),
    ]
    for lang, text, expected_lemma in cases:
        model = _local_model(lang)
        if model is None:
            print(f"    (skipped {lang}: no local UDPipe model)")
            continue
        try:
            from ufal.udpipe import Model  # noqa: F401
        except ImportError:
            print("    (skipped: ufal.udpipe not installed)")
            continue

        from scptm.nlp import load_udpipe_pipeline

        nlp = load_udpipe_pipeline(lang, model_path=model, verbose=False)
        tokens = list(nlp(text))
        assert tokens, f"UDPipe returned no tokens for '{lang}'"
        lemmas = {t.lemma_.lower() for t in tokens}
        deps = {t.dep_ for t in tokens}
        assert expected_lemma in lemmas, (lang, lemmas)
        assert "root" in deps, (lang, deps)
        # Every token must point at a token in the same document.
        for token in tokens:
            assert token.head in tokens


if __name__ == "__main__":          # pragma: no cover - manual runner
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:               # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{'FAILED' if failures else 'OK'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
