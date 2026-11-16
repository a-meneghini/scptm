"""
scptm/cli.py
------------
Command line front-end for the neural SCPTM: ``scptm fit`` and ``scptm topics``.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import SCPTMConfig
from .model import SCPTM


def _read_corpus(path: Path, mode: str) -> List[str]:
    text = path.read_text(encoding="utf-8")
    if mode == "blank":
        docs = [c.strip() for c in text.split("\n\n")]
    else:
        docs = [line.strip() for line in text.splitlines()]
    return [d for d in docs if d]


def cmd_fit(args) -> int:
    docs = _read_corpus(Path(args.corpus), args.separator)
    if not docs:
        print("error: empty corpus", file=sys.stderr)
        return 2
    cfg = SCPTMConfig(
        lang=args.lang,
        num_topics=args.topics,
        epochs=args.epochs,
        min_df=args.min_df,
        max_features=args.max_features,
        device=args.device,
        lemmatizer=args.parser,
        udpipe_model_path=args.model_path,
    )
    model = SCPTM(config=cfg)
    model.fit(docs)
    for row in model.get_topic_info(top_k=10).itertuples():
        print(f"[{row.topic_id:>2}] n={row.size:>5}  {row.keywords}")
    if args.out:
        model.save(args.out)
        print(f"\nmodel -> {args.out}")
    return 0


def cmd_topics(args) -> int:
    model = SCPTM.load(args.model)
    for row in model.get_topic_info(top_k=args.top_k).itertuples():
        print(f"[{row.topic_id:>2}] n={row.size:>5}  {row.keywords}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="scptm",
        description="Structural Contextual Probabilistic Topic Model.",
    )
    sub = root.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a model on a text file")
    fit.add_argument("corpus")
    fit.add_argument("--separator", choices=("line", "blank"), default="line")
    fit.add_argument("--topics", type=int, default=20)
    fit.add_argument("--lang", choices=("eng", "ita"), default="eng")
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--min-df", dest="min_df", type=int, default=5)
    fit.add_argument("--max-features", dest="max_features", type=int, default=15_000)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--parser", choices=("spacy", "udpipe"), default="spacy")
    fit.add_argument("--model-path", dest="model_path", default=None)
    fit.add_argument("--out", default=None)
    fit.set_defaults(func=cmd_fit)

    topics = sub.add_parser("topics", help="print the topics of a saved model")
    topics.add_argument("model")
    topics.add_argument("--top-k", dest="top_k", type=int, default=10)
    topics.set_defaults(func=cmd_topics)
    return root


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
