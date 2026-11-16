# Releasing

Two **independent** distributions live in this repository:

| package | import | PyPI name | version source |
|---|---|---|---|
| `scptm` | `scptm` | `scptm` | `scptm/__init__.py::__version__` |
| `scptm-bow` | `scptm_bow` | `scptm-bow` | `packages/scptm-bow/pyproject.toml` |

`scptm_bow` does **not** depend on `scptm`, and `scptm`'s sdist prunes
`packages/scptm-bow/` (a separate directory, outside the scptm sdist root). Verified: the `scptm` sdist contains zero
`scptm-bow` entries.

> PyPI normalises names: `scptm-bow` and `scptm_bow` are the *same* project. The
> first upload registers the name — it cannot be reclaimed or reused after that.

## 1. Bump the version

```bash
# scptm
$EDITOR packages/scptm/src/scptm/__init__.py            # __version__ = "x.y.z"
$EDITOR packages/scptm/CITATION.cff  # version: x.y.z

# scptm-bow
$EDITOR packages/scptm-bow/pyproject.toml     # version = "x.y.z"
$EDITOR packages/scptm-bow/CITATION.cff       # version: x.y.z
```

## 2. Green before shipping

```bash
for f in packages/scptm/tests/test_*.py; do python "$f" || exit 1; done
for f in packages/scptm-bow/tests/test_*.py; do python "$f" || exit 1; done
python conformance/make_fixture.py   # must not change the fixture
git diff --exit-code conformance/    # if it did, something is non-deterministic
```

## 3. Build and inspect

```bash
rm -rf dist
python -m build ./packages/scptm --outdir dist
python -m build ./packages/scptm-bow --outdir dist-bow
twine check dist/* dist-bow/*

# the sdist must never ship the other package
tar tzf dist/scptm-*.tar.gz | grep -c 'scptm_bow'    # must print 0
```

## 4. Upload — TestPyPI first

Always rehearse on TestPyPI: a real upload cannot be undone.

```bash
# TestPyPI
twine upload --repository testpypi dist/*
twine upload --repository testpypi dist-bow/*

# smoke test in a clean environment
python -m venv /tmp/tp && /tmp/tp/bin/pip install \
  -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
  scptm scptm-bow
/tmp/tp/bin/python -c "import scptm, scptm_bow; print(scptm.__version__, scptm_bow.__version__)"
```

## 5. Upload to PyPI

Two options.

**A. Trusted Publishing (recommended, no secrets).** Create a GitHub *release*
(or run the workflow manually) — `.github/workflows/publish.yml` builds, checks
and publishes both packages via OIDC. One-time setup on PyPI →
*Account → Publishing*:

* `scptm` — add a publisher for `a-meneghini/scptm`, workflow `publish.yml`,
  environment `pypi`.
* `scptm-bow` — add a **pending** publisher with the same coordinates; the
  project is created on the first successful upload.

**B. API token (manual).**

```bash
twine upload dist/*
twine upload dist-bow/*
# username: __token__   password: pypi-...
```

## 6. Tag

```bash
git tag -a v0.2.2 -m "scptm 0.2.2 + scptm-bow 0.2.0"
git push origin v0.2.2
```
