"""
Retrieval relevance evaluator for Self-RAG.

Checks the quality of retrieved documents and decides
whether another retrieval attempt is required.
"""


def retrieval_evaluator_node(state):

    retrieved_docs = state.get(
        "retrieved_text",
        []
    )


    if not retrieved_docs:

        print(
            "No documents retrieved. Triggering re-query."
        )

        return {
            "needs_requery": True,
            "first_score": 0.0,
            "retrieval_relevance_score": 0.0
        }


    # Get top Qdrant similarity score

    score = retrieved_docs[0].get(
        "score",
        0.0
    )


    print(
        f"Retrieval relevance score: {score}"
    )


    threshold = 0.70


    if score < threshold:

        print(
            "Low relevance detected. Rewriting query..."
        )

        return {

            "needs_requery": True,

            "first_score": score,

            "retrieval_relevance_score": score,

        }


    print(
        "Retrieval quality is good."
    )


    return {

        "needs_requery": False,

        "first_score": score,

        "retrieval_relevance_score": score,

    }