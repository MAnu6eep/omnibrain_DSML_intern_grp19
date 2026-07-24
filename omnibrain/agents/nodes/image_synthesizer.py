from typing import Dict, List


def format_image_context(
    images: List[Dict]
) -> Dict:
    """
    Formats retrieved image results into:

    1. formatted_text:
       Used inside the LLM prompt.

    2. image_records:
       Used by FastAPI / Streamlit UI.

    Args:
        images:
            List returned by search_images().

    Returns:
        Dictionary containing formatted text and
        structured image records.
    """

    if not images:
        return {
            "formatted_text": "No relevant figures were retrieved.",
            "image_records": [],
        }

    lines = ["Visual Context", ""]

    image_records = []

    for image in images:

        page_number = image.get("page_number")
        caption = image.get("caption") or "No caption available"
        image_path = image.get("image_path")

        lines.append(f"Image Page {page_number}")
        lines.append(f"Caption: {caption}")
        lines.append("")

        image_records.append(
            {
                "image_path": image_path,
                "page_number": page_number,
                "caption": caption,
            }
        )

    return {
        "formatted_text": "\n".join(lines).strip(),
        "image_records": image_records,
    }