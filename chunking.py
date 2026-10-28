"""Chunking strategies. Each returns a list of (doc_id, chunk_text)."""
import re


def fixed(docs, size=300, overlap=0):
    """Fixed-size character windows, optionally overlapping."""
    out = []
    for d in docs:
        text, step = d["text"], max(size - overlap, 1)
        for start in range(0, len(text), step):
            out.append((d["id"], text[start:start + size]))
            if start + size >= len(text):
                break
    return out


def paragraph(docs):
    """One chunk per paragraph, with the document title prepended for context."""
    return [
        (d["id"], f"{d['title']}. {p.strip()}")
        for d in docs
        for p in d["text"].split("\n\n")
        if p.strip()
    ]


def sentences(docs, per_chunk=2):
    """Groups of N sentences."""
    out = []
    for d in docs:
        sents = re.split(r"(?<=[.!?])\s+", d["text"].replace("\n\n", " "))
        for i in range(0, len(sents), per_chunk):
            out.append((d["id"], " ".join(sents[i:i + per_chunk])))
    return out


STRATEGIES = {
    "fixed-300": lambda docs: fixed(docs, 300),
    "fixed-600+100": lambda docs: fixed(docs, 600, 100),
    "sentences-2": lambda docs: sentences(docs, 2),
    "paragraph+title": paragraph,
}
