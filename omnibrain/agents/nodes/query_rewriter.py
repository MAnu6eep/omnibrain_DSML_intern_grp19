"""
Query Rewriter for Self-RAG

Rewrites user queries when retrieval quality is poor so that
a second vector search has a better chance of finding relevant
documents.
"""

from typing import Dict, List


class QueryRewriter:
    """Generates an improved search query."""

    def __init__(self):
        pass

    def rewrite(
        self,
        query: str,
        retrieved_docs: List[Dict],
    ) -> str:
        """
        Rewrite the original query based on failed retrieval results.

        Args:
            query: Original user query.
            retrieved_docs: Documents returned by the first retrieval attempt.

        Returns:
            Improved search query.
        """

        query = query.strip()

        # No retrieval results → slightly broaden the search
        if not retrieved_docs:
            return f"{query} detailed explanation"

        # Collect document titles/sources
        sources = []

        for doc in retrieved_docs:
            source = doc.get("document") or doc.get("source")

            if source and source not in sources:
                sources.append(source)

        # If we know the document name, include it
        if sources:
            rewritten_query = (
                f"{query} related to {' '.join(sources)}"
            )
        else:
            rewritten_query = f"{query} detailed explanation"

        return rewritten_query

    def should_retry(
        self,
        relevance_score: float,
        threshold: float = 0.70,
    ) -> bool:
        """
        Decide whether retrieval should be retried.

        Args:
            relevance_score: Best similarity score.
            threshold: Minimum acceptable similarity.

        Returns:
            True if another retrieval attempt should be made.
        """

        return relevance_score < threshold