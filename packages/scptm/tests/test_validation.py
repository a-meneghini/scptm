"""
tests/test_validation.py
========================
Tests for argument/config validation.

Two regressions are covered here:

1. ``SCPTMConfig`` used to validate with ``assert``, which is silently stripped
   under ``python -O``: an invalid config would then run and fail much later in
   a confusing place.  Validation now raises ``ValueError``.
2. ``SCPTM(config=cfg, num_topics=25)`` used to discard ``num_topics`` without a
   word, because ``config or SCPTMConfig(**kwargs)`` short-circuited.  Passing
   both now raises ``TypeError``.

Also covers the vocabulary token pattern, which used to drop every accented
Italian word ("città", "però", "più", …).

No model is loaded and no network is used.

Run with pytest::

    pytest tests/test_validation.py -v

or directly (no pytest required)::

    python tests/test_validation.py
"""

import re
import traceback

from scptm import SCPTM
from scptm.config import SCPTMConfig
from scptm.graph import VOCAB_TOKEN_PATTERN
from scptm.nlp import _normalise_lang


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception as other:               # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}"
        )
    return False


# ---------------------------------------------------------------------------
# Config validation: ValueError, not assert
# ---------------------------------------------------------------------------

def test_config_rejects_invalid_fields_with_value_error():
    cases = [
        {"graph_mode": "bogus"},
        {"kl_strategy": "bogus"},
        {"bow_normalization": "bogus"},
        {"keyword_method": "bogus"},
        {"lemmatizer": "stanza"},
        {"device": "tpu"},
        {"lang": "deu"},
    ]
    for kwargs in cases:
        assert _raises(ValueError, SCPTMConfig, **kwargs), (
            f"SCPTMConfig(**{kwargs}) must raise ValueError (not AssertionError, "
            "which python -O would strip)"
        )


def test_config_accepts_supported_langs():
    for lang in ("eng", "ita"):
        assert SCPTMConfig(lang=lang).lang == lang


# ---------------------------------------------------------------------------
# SCPTM(config=...) vs kwargs
# ---------------------------------------------------------------------------

def test_scptm_rejects_config_plus_kwargs():
    cfg = SCPTMConfig(num_topics=10)
    assert _raises(TypeError, SCPTM, config=cfg, num_topics=25)


def test_scptm_accepts_config_only():
    cfg = SCPTMConfig(num_topics=7)
    assert SCPTM(config=cfg).config.num_topics == 7


def test_scptm_accepts_kwargs_only():
    assert SCPTM(num_topics=7).config.num_topics == 7


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------

def test_normalise_lang_accepts_known_codes():
    assert _normalise_lang("ita") == "ita"
    assert _normalise_lang("ENG") == "eng"      # case-insensitive
    assert _normalise_lang(None) == "eng"       # documented default


def test_normalise_lang_rejects_unknown_code():
    # Used to fall back to English silently, turning a typo into an English run.
    assert _raises(ValueError, _normalise_lang, "deu")


# ---------------------------------------------------------------------------
# Vocabulary token pattern (accented words)
# ---------------------------------------------------------------------------

def test_token_pattern_keeps_accented_words():
    text = "città però più società università machine"
    found = re.findall(VOCAB_TOKEN_PATTERN, text)
    for word in ("città", "però", "più", "società", "università", "machine"):
        assert word in found, f"'{word}' was dropped from the vocabulary"


def test_token_pattern_rejects_digits_and_short_tokens():
    found = re.findall(VOCAB_TOKEN_PATTERN, "covid19 123 a x4 _privato")
    assert "123" not in found
    assert "covid19" not in found
    assert "a" not in found                      # needs at least 2 letters


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
