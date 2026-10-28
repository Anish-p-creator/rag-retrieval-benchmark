"""Benchmark chunking strategies x retrievers on a labelled QA set.

  python benchmark.py              # includes dense retrieval if sentence-transformers is available
  python benchmark.py --no-dense   # lexical retrievers only, runs in seconds

Metrics are computed at document level: a query is a hit if any retrieved chunk
comes from the gold document.
"""
import argparse
import json
from functools import partial

from chunking import STRATEGIES
from retrievers import BM25, TFIDF, Dense, Hybrid


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_ranking(chunk_ranking, chunk_doc_ids):
    """Collapse a chunk ranking into a ranking of unique documents."""
    seen, docs = set(), []
    for idx in chunk_ranking:
        d = chunk_doc_ids[idx]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def evaluate(retriever, chunk_doc_ids, queries):
    r1 = r3 = rr = 0.0
    for q in queries:
        docs = doc_ranking(retriever.rank(q["query"]), chunk_doc_ids)
        rank = docs.index(q["gold"]) + 1 if q["gold"] in docs else None
        r1 += rank == 1
        r3 += rank is not None and rank <= 3
        rr += 1.0 / rank if rank else 0.0
    n = len(queries)
    return r1 / n, r3 / n, rr / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus.jsonl")
    p.add_argument("--queries", default="queries.jsonl")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--out", default="results.md")
    args = p.parse_args()

    docs, queries = load_jsonl(args.corpus), load_jsonl(args.queries)

    retrievers = [BM25, TFIDF]
    if not args.no_dense:
        try:
            Dense(["warm-up"])
            retrievers += [Dense, partial(Hybrid, members=[BM25, Dense])]
        except Exception as e:
            print(f"[warn] dense retrieval unavailable ({e.__class__.__name__}); lexical only")
    if len(retrievers) == 2:
        retrievers.append(partial(Hybrid, members=[BM25, TFIDF]))

    rows = []
    for strat_name, strat in STRATEGIES.items():
        chunks = strat(docs)
        ids, texts = [c[0] for c in chunks], [c[1] for c in chunks]
        for make in retrievers:
            r = make(texts)
            r1, r3, mrr = evaluate(r, ids, queries)
            rows.append((strat_name, len(chunks), r.name, r1, r3, mrr))

    rows.sort(key=lambda r: (-r[5], -r[3]))
    lines = [
        f"Corpus: {len(docs)} documents, {len(queries)} labelled queries\n",
        "| Chunking | Chunks | Retriever | Recall@1 | Recall@3 | MRR |",
        "|---|---:|---|---:|---:|---:|",
    ] + [f"| {s} | {n} | {name} | {a:.2f} | {b:.2f} | {c:.3f} |" for s, n, name, a, b, c in rows]
    table = "\n".join(lines)
    print(table)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Benchmark results\n\n" + table + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
