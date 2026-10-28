"""Retrievers. Each takes a list of chunk texts and exposes rank(query) -> chunk indices, best first."""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25:
    name = "BM25"

    def __init__(self, texts):
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query):
        return list(np.argsort(-self.index.get_scores(tokenize(query))))


class TFIDF:
    name = "TF-IDF"

    def __init__(self, texts):
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform(texts)

    def rank(self, query):
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        return list(np.argsort(-scores))


class Dense:
    name = "Dense (MiniLM)"
    _model = None

    def __init__(self, texts):
        if Dense._model is None:
            from sentence_transformers import SentenceTransformer
            Dense._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.vectors = Dense._model.encode(texts, normalize_embeddings=True)

    def rank(self, query):
        q = Dense._model.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.vectors @ q)))


class Hybrid:
    """Reciprocal Rank Fusion of several retrievers."""

    def __init__(self, texts, members, k=60):
        self.members = [m(texts) for m in members]
        self.k = k
        self.name = "Hybrid RRF (" + " + ".join(m.name for m in self.members) + ")"

    def rank(self, query):
        scores = {}
        for m in self.members:
            for r, idx in enumerate(m.rank(query), start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.k + r)
        return sorted(scores, key=scores.get, reverse=True)
