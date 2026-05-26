"""
Local, offline embedding for the WhatsApp RAG.

Strategy:
  1. If OPENAI_API_KEY is set AND --embedder=openai is passed, use OpenAI
     text-embedding-3-small (1536-d) — best semantic quality.
  2. Otherwise, fit a TF-IDF + TruncatedSVD model on the corpus the first
     time and pickle it next to the DB. Subsequent runs reuse it.

The TF-IDF+SVD path needs no network and is well suited to evidence-style
retrieval (payment refusal, voice note around date X, etc.) because exact
terms matter a lot in transcripts.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


VECTORIZER_FILE = "tfidf_svd.pkl"
EMBED_DIM = 256  # SVD output dimension


class TfidfSvdEmbeddingFunction(EmbeddingFunction):
    """TF-IDF + TruncatedSVD. Must be fit() on the corpus before .embed()."""

    name_ = "tfidf-svd-256"

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.model_dir / VECTORIZER_FILE
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        if self.path.exists():
            self._load()

    # Required by chromadb >=0.5: name() returns a stable identifier
    @staticmethod
    def name() -> str:
        return "tfidf-svd-256"

    def _load(self):
        with open(self.path, "rb") as f:
            payload = pickle.load(f)
        self.vectorizer = payload["vectorizer"]
        self.svd = payload["svd"]

    def fit(self, corpus: List[str]):
        if not corpus:
            raise ValueError("Empty corpus")
        # Tokenizer keeps alnum tokens incl. digits (dates) and unicode words.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            max_features=100_000,
        )
        X = self.vectorizer.fit_transform(corpus)
        n_comp = min(EMBED_DIM, X.shape[1] - 1, X.shape[0] - 1)
        n_comp = max(n_comp, 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(X)
        with open(self.path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("Embedder not fit; call fit(corpus) first.")
        X = self.vectorizer.transform(list(input))
        emb = self.svd.transform(X).astype(np.float32)
        # L2 normalize for cosine
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        return emb.tolist()


def build_embedder(model_dir: str | Path, kind: str = "tfidf") -> EmbeddingFunction:
    if kind == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set; pass --embedder tfidf instead")
        return OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-3-small"
        )
    return TfidfSvdEmbeddingFunction(model_dir)
