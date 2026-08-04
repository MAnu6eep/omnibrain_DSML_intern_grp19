"""
Retrieval Grader for Self-RAG

Evaluates whether retrieved documents from Qdrant are relevant
based on cosine similarity scores.
"""

from typing import List, Dict

SIMILARITY_THRESHOLD = 0.70


class RetrievalGrader:
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold

    def grade(
        self,
        query: str,
        retrieved_docs: List[Dict],
    ) -> Dict:
        """
        Evaluate retrieved documents.

        Args:
            query: User question
            retrieved_docs: List returned from search_text_chunks()

        Returns:
            Dictionary containing evaluation results.
        """

        # No documents retrieved
        if not retrieved_docs:
            return {
                "query": query,
                "is_relevant": False,
                "confidence": 0.0,
                "best_score": 0.0,
                "reason": "No documents retrieved."
            }

        # Highest similarity score
        best_score = max(doc.get("score", 0.0) for doc in retrieved_docs)

        is_relevant = best_score >= self.threshold

        return {
            "query": query,
            "is_relevant": is_relevant,
            "confidence": round(best_score, 4),
            "best_score": round(best_score, 4),
            "threshold": self.threshold,
            "reason": (
                "Relevant context found."
                if is_relevant
                else "Retrieved context below similarity threshold."
            ),
        }