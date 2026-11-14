from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)


def _chunk(text: str, words_per_chunk: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


class SemanticScorer:
    def __init__(self, model_name: str, use_transformers: bool = True):
        self.model_name = model_name
        self.backend = "tfidf"
        self.note = ""
        self._model = None

        if use_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "transformers"
            except Exception as exc:
                self.note = f"Transformer model unavailable ({type(exc).__name__}); using TF-IDF fallback."
                log.warning(self.note)
        else:
            self.note = "Transformers disabled by configuration; using TF-IDF."

    def score(self, job_text: str, cv_texts: list[str]) -> list[float]:
        if not cv_texts:
            return []
        if self.backend == "transformers":
            try:
                return self._score_transformers(job_text, cv_texts)
            except Exception as exc:
                self.note = f"Transformer scoring failed ({type(exc).__name__}); used TF-IDF."
                log.warning(self.note)
                self.backend = "tfidf"
        return self._score_tfidf(job_text, cv_texts)

    def _score_transformers(self, job_text: str, cv_texts: list[str]) -> list[float]:
        jd_chunks = _chunk(job_text)
        jd_emb = self._model.encode(jd_chunks, normalize_embeddings=True, show_progress_bar=False)

        scores = []
        for cv in cv_texts:
            cv_chunks = _chunk(cv)
            cv_emb = self._model.encode(cv_chunks, normalize_embeddings=True, show_progress_bar=False)
            sim = np.asarray(jd_emb) @ np.asarray(cv_emb).T
            coverage = float(sim.max(axis=1).mean())
            scores.append(float(np.clip((coverage - 0.15) / 0.50, 0.0, 1.0)))
        return scores

    def _score_tfidf(self, job_text: str, cv_texts: list[str]) -> list[float]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        docs = [job_text] + cv_texts
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w+#.]{2,}\b",
        )
        try:
            matrix = vec.fit_transform(docs)
        except ValueError:
            return [0.0] * len(cv_texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        return [float(np.clip(s / 0.40, 0.0, 1.0)) for s in sims]


def explain_top_terms(job_text: str, cv_text: str, n: int = 8) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][\w+#.-]{2,}\b")
    try:
        m = vec.fit_transform([job_text, cv_text]).toarray()
    except ValueError:
        return []
    terms = np.array(vec.get_feature_names_out())
    overlap = np.minimum(m[0], m[1])
    idx = overlap.argsort()[::-1][:n]
    return [str(terms[i]) for i in idx if overlap[i] > 0]
