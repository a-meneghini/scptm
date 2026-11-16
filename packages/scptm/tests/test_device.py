"""
tests/test_device.py
====================
Tests for device resolution (``auto`` / ``cpu`` / ``cuda`` / ``mps``).

Environment-agnostic: the CUDA/MPS assertions are conditioned on what the
running machine actually has.  No model is loaded and no network is used.

Run with pytest::

    pytest tests/test_device.py -v

or directly (no pytest required)::

    python tests/test_device.py
"""

import traceback

import torch

from scptm.config import DEVICES, SCPTMConfig, resolve_device


def _cuda() -> bool:
    return torch.cuda.is_available()


def _mps() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:                       # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_prefers_accelerator():
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")
    if _cuda():
        assert dev.type == "cuda"
    elif _mps():
        assert dev.type == "mps"
    else:
        assert dev.type == "cpu"


def test_resolve_device_explicit_cuda():
    if _cuda():
        assert resolve_device("cuda").type == "cuda"
    else:
        raised = False
        try:
            resolve_device("cuda")
        except RuntimeError:
            raised = True
        assert raised, "explicit cuda without CUDA must raise RuntimeError"


def test_resolve_device_explicit_mps():
    if _mps():
        assert resolve_device("mps").type == "mps"
    else:
        raised = False
        try:
            resolve_device("mps")
        except RuntimeError:
            raised = True
        assert raised, "explicit mps without MPS must raise RuntimeError"


def test_resolve_device_invalid_raises():
    raised = False
    try:
        resolve_device("tpu")
    except ValueError:
        raised = True
    assert raised, "an unknown device string must raise ValueError"


# ---------------------------------------------------------------------------
# SCPTMConfig.device
# ---------------------------------------------------------------------------

def test_config_accepts_all_devices():
    for dev in DEVICES:
        assert SCPTMConfig(device=dev).device == dev
    assert SCPTMConfig().device == "auto"      # default


def test_config_rejects_invalid_device():
    raised = False
    try:
        SCPTMConfig(device="tpu")
    except ValueError:
        raised = True
    assert raised, "SCPTMConfig must reject an unsupported device (ValueError)"


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
