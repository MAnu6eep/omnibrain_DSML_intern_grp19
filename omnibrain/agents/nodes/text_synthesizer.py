"""
Text Synthesizer

Formats retrieved text chunks into a single context string
for downstream LLM synthesis.
"""


def format_text_context(results: list[dict]) -> str:
    """
    Merge retrieved text chunks into one formatted context.

    Args:
        results: Output from search_text_chunks()

    Returns:
        Formatted context string.
    """

    if not results:
        return "No relevant context found."

    formatted_chunks = []

    for index, item in enumerate(results, start=1):

        formatted_chunks.append(
            f"""
==================================================
Chunk {index}

Chunk ID : {item.get("chunk_id", "Unknown")}
Document : {item.get("document", "Unknown Document")}
Page     : {item.get("page", "Unknown")}
Score    : {item.get("score", 0)}

Text:
{item.get("text", "")}
""".strip()
        )

    return "\n\n".join(formatted_chunks)


if __name__ == "__main__":

    sample_results = [
        {
            "chunk_id": "chunk_001",
            "document": "AI_Book.pdf",
            "page": 5,
            "score": 0.95,
            "text": (
                "Retrieval-Augmented Generation combines retrieval "
                "with language models."
            ),
        },
        {
            "chunk_id": "chunk_002",
            "document": "AI_Book.pdf",
            "page": 6,
            "score": 0.91,
            "text": "Vector databases enable semantic similarity search.",
        },
    ]

    context = format_text_context(sample_results)

    print(context)
