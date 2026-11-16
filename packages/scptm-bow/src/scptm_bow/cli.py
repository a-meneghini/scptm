"""
scptm_bow/cli.py
----------------
Command line front-end: ``scptm-bow fit`` and ``scptm-bow topics``.

Kept deliberately small — it exists so a corpus can be turned into a model
without writing Python, and so the reproduction steps in a paper are one line
rather than a notebook cell.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import BowConfig
from .model import BowSCPTM


def _read_corpus(path: Path, mode: str) -> List[str]:
    text = path.read_text(encoding="utf-8")
    if mode == "blank":
        docs = [chunk.strip() for chunk in text.split("\n\n")]
    else:
        docs = [line.strip() for line in text.splitlines()]
    return [d for d in docs if d]


def _build(args) -> BowSCPTM:
    cfg = BowConfig(
        num_topics=args.topics,
        lang=args.lang,
        parser_backend=args.parser,
        udpipe_model_path=args.model_path,
        min_df=args.min_df,
        max_features=args.max_features,
        divergence=args.divergence,
        nmf_max_iter=args.iterations,
        topic_stopwords=args.stopwords.split(",") if args.stopwords else None,
    )
    return BowSCPTM(config=cfg)


def cmd_fit(args) -> int:
    corpus = _read_corpus(Path(args.corpus), args.separator)
    if not corpus:
        print("error: empty corpus", file=sys.stderr)
        return 2
    model = _build(args)
    model.fit(corpus, cache_path=args.cache, verbose=True)

    print(f"\nvocabulary: {model.parsed.n_vocab:,} lemmas")
    stats = model.evaluate()
    for key in ("c_v", "npmi", "diversity", "exclusivity", "redundancy"):
        value = stats.get(key)
        if value is not None:
            print(f"  {key:<12} {value:.4f}")

    print(f"\n{'topic':>6}  words")
    for topic in model.get_topic_info(top_k=10):
        print(f"{topic['topic_id']:>6}  {topic['top_words']}")

    if args.out:
        model.save(args.out)
        print(f"\nmodel -> {args.out}")
    if args.figure:
        model.visualize_2d(top_k=10, save_path=args.figure)
    return 0


def cmd_topics(args) -> int:
    model = BowSCPTM.load(args.model)
    extra = args.stopwords.split(",") if args.stopwords else None
    for topic in model.get_topic_info(top_k=args.top_k, extra_stopwords=extra):
        print(f"[{topic['topic_id']:>2}] n={topic['size']:>5}  {topic['top_words']}")
        if topic.get("mwes"):
            print(f"      MWE: {topic['mwes']}")
    return 0


def cmd_evaluate(args) -> int:
    model = BowSCPTM.load(args.model)
    corpus = _read_corpus(Path(args.corpus), args.separator)
    print(f"held-out perplexity: {model.heldout_perplexity(corpus):.4f}")
    if args.json:
        print(json.dumps(model.evaluate(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="scptm-bow",
        description="Structure-aware topic modelling over a lemmatised bag of words.",
    )
    sub = root.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a model on a text file")
    fit.add_argument("corpus", help="one document per line (or per empty-line block)")
    fit.add_argument("--separator", choices=("line", "blank"), default="line")
    fit.add_argument("--topics", type=int, default=20)
    fit.add_argument("--lang", choices=("eng", "ita"), default="eng")
    fit.add_argument("--parser", choices=("spacy", "udpipe"), default="spacy")
    fit.add_argument("--model-path", dest="model_path", default=None)
    fit.add_argument("--min-df", dest="min_df", type=int, default=5)
    fit.add_argument("--max-features", dest="max_features", type=int, default=15_000)
    fit.add_argument("--divergence", choices=("kl", "frobenius"), default="kl")
    fit.add_argument("--iterations", type=int, default=200)
    fit.add_argument("--stopwords", default=None, help="comma-separated extra stop words")
    fit.add_argument("--cache", default=None)
    fit.add_argument("--out", default=None, help="save the fitted model here")
    fit.add_argument("--figure", default=None, help="write the 2D semantic figure here")
    fit.set_defaults(func=cmd_fit)

    topics = sub.add_parser("topics", help="print the topics of a saved model")
    topics.add_argument("model")
    topics.add_argument("--top-k", dest="top_k", type=int, default=10)
    topics.add_argument("--stopwords", default=None)
    topics.set_defaults(func=cmd_topics)

    ev = sub.add_parser("evaluate", help="held-out perplexity and metrics")
    ev.add_argument("model")
    ev.add_argument("corpus")
    ev.add_argument("--separator", choices=("line", "blank"), default="line")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_evaluate)
    return root


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
