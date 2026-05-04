import os
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

class CorpusRetriever:
    """BM25 index over the full SciFact corpus (title + abstract sentences)."""

    def __init__(self, corpus: dict):
        print("Building BM25 index over full corpus...")
        self.doc_ids   = list(corpus.keys())
        self.corpus    = corpus
        # Tokenize title + all abstract sentences concatenated
        tokenized = [
            (corpus[did]["title"] + " " + " ".join(corpus[did]["abstract"])).lower().split()
            for did in self.doc_ids
        ]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  Index built ({len(self.doc_ids)} docs).")

    def retrieve(self, query: str, k: int = int(os.environ.get("TOP_K_ABSTRACTS"))) -> list[dict]:
        """Return top-k docs sorted by BM25 score (highest first)."""
        scores  = self.bm25.get_scores(query.lower().split())
        indices = np.argsort(-scores)[:k]
        results = []
        for idx in indices:
            did = self.doc_ids[idx]
            results.append({
                "doc_id":    did,
                "title":     self.corpus[did]["title"],
                "abstract":  self.corpus[did]["abstract"],
                "bm25_score": float(scores[idx]),
            })
        return results

    def sentence_scores(self, claim: str, sentences: list[str]) -> list[float]:
        """
        Score each sentence against the claim.
        Build BM25 over ALL sentences in the abstract together so IDF is
        computed across the full set (avoids degenerate single-doc IDF).
        Fall back to raw token-overlap count when the abstract is tiny.
        """
        if not sentences:
            return []
        q_tokens = claim.lower().split()
        tokenized = [s.lower().split() for s in sentences]

        if len(sentences) >= 3:
            bm25   = BM25Okapi(tokenized)
            scores = [float(s) for s in bm25.get_scores(q_tokens)]
        else:
            # With < 3 sentences BM25 IDF is unreliable; use token overlap
            q_set  = set(q_tokens)
            scores = [float(len(q_set & set(toks))) for toks in tokenized]

        return scores