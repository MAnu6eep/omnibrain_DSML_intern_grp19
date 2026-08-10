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
            "text": "Artificial Intelligence is transforming healthcare."

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

def format_visual_context(results: list[dict]) -> str:
    """
    Format VLM extracted image/chart context for downstream synthesis.

    Args:
        results: List of VLM extraction dictionaries.

    Returns:
        Formatted visual context string.
    """

    if not results:
        return ""

    formatted = []

    for index, item in enumerate(results, start=1):

        values = item.get("values", [])

        value_lines = []

        for value in values:
            value_lines.append(
                f"{value.get('label', '')} = {value.get('value', '')}"
            )

        formatted.append(
            f"""
==================================================
Vision Context {index}

Document : {item.get("source", "Unknown")}
Page     : {item.get("page_number", "Unknown")}

Caption:
{item.get("caption", "")}

Chart Type:
{item.get("chart_type", "")}

Title:
{item.get("title", "")}

Extracted Values:

{chr(10).join(value_lines)}

Summary:

{item.get("summary", "")}
""".strip()
        )

    return "\n\n".join(formatted)

def merge_contexts(
    text_results: list[dict],
    visual_results: list[dict],
) -> str:
    """
    Merge text and visual context into a single prompt.
    """

    text_context = format_text_context(text_results)
    visual_context = format_visual_context(visual_results)

    if visual_context:
        return f"{text_context}\n\n{visual_context}"

    return text_context
