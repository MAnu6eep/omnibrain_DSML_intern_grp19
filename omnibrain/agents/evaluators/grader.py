"""
omnibrain/agents/evaluators/grader.py
Self-RAG Grader Evaluator: Combines LLM relevance grading
and Cosine Similarity thresholding.
"""

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from omnibrain.agents.llm import get_llm_response
from omnibrain.agents.state.state import AgentState

SIMILARITY_THRESHOLD = 0.70


# 1. Pydantic schema for structured grading response
class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    explanation: str = Field(description="Brief reasoning for the grading decision")


# 2. Define the Grader Prompt
GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict relevance grader assessing whether retrieved "
                "document context is relevant to a user query.\n\n"
                "Rule: If the context contains keywords, semantic intent, or data "
                "directly related to answering the user question, grade it as 'yes'.\n"
                "Otherwise, grade it as 'no'. Be objective and do not assume facts "
                "not present in the context."
            ),
        ),
        ("human", "Retrieved Context:\n{context}\n\nUser Question: {question}"),
    ]
)


# 3. Main LangGraph Node Function
def grader_node(state: AgentState) -> dict:
    """
    Evaluates retrieved text/image context in AgentState for relevance.
    Updates `retrieval_relevance_score` and appends to `thought_process`.
    """
    # Extract user question from state messages
    messages = state.get("messages", [])
    question = ""
    if messages:
        last_msg = messages[-1]
        question = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    retrieved_text = state.get("retrieved_text", [])
    retrieved_images = state.get("retrieved_images", [])

    # Combine text snippets and image captions into context block
    context_snippets = []
    for item in retrieved_text:
        context_snippets.append(item.get("text", item.get("content", "")))
    for img in retrieved_images:
        if img.get("caption"):
            context_snippets.append(f"[Image Caption]: {img.get('caption')}")

    combined_context = "\n---\n".join(context_snippets)

    # Edge Case: If no context was retrieved at all
    if not combined_context.strip():
        thought = {
            "agent": "Grader Agent",
            "action": "Graded context as 'no' (Zero chunks/images retrieved).",
        }
        return {"retrieval_relevance_score": "no", "thought_process": [thought]}

    # Execute grading prompt via get_llm_response
    try:
        prompt_input = GRADER_PROMPT.format(question=question, context=combined_context)
        response_text, provider = get_llm_response(prompt_input)

        # Extract score ("yes" or "no") from LLM response
        response_lower = response_text.lower()
        score = "yes" if "yes" in response_lower else "no"
        explanation = response_text.strip()
    except Exception as e:
        score = "yes" if len(combined_context) > 100 else "no"
        explanation = f"Fallback evaluation executed due to: {str(e)}"

    thought = {
        "agent": "Grader Agent",
        "action": (
            f"Graded retrieval relevance as '{score.upper()}'. Reason: {explanation}"
        ),
    }

    return {"retrieval_relevance_score": score, "thought_process": [thought]}


# 4. Cosine Similarity Threshold Grader Class
class RetrievalGrader:
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold

    def grade(
        self,
        query: str,
        retrieved_docs: List[Dict],
    ) -> Dict:
        """
        Evaluate retrieved documents based on cosine similarity scores.
        """
        if not retrieved_docs:
            return {
                "query": query,
                "is_relevant": False,
                "confidence": 0.0,
                "best_score": 0.0,
                "reason": "No documents retrieved.",
            }

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
