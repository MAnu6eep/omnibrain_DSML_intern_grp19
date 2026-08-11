from qdrant_client import QdrantClient

from omnibrain.vectorstore.collections import (
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
)
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper


def verify_collection(collection_name, required_fields):
    print("\n" + "=" * 60)
    print(f"Checking collection: {collection_name}")
    print("=" * 60)

    client = QdrantClientWrapper().client()

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        print("❌ No points found in collection.")
        return False

    all_valid = True

    for point in points:
        payload = point.payload or {}

        print(f"\nPoint ID: {point.id}")

        missing_fields = [
            field
            for field in required_fields
            if field not in payload
        ]

        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
            all_valid = False
        else:
            print("✅ Required metadata present")

        for field in required_fields:
            print(
                f"   {field}: "
                f"{payload.get(field)!r}"
            )

    return all_valid


def main():
    text_valid = verify_collection(
        TEXT_COLLECTION,
        [
            "chunk_id",
            "source",
            "page_number",
            "document_id",
        ],
    )

    image_valid = verify_collection(
        IMAGE_COLLECTION,
        [
            "image_id",
            "source",
            "page_number",
            "document_id",
        ],
    )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if text_valid and image_valid:
        print("✅ Citation metadata verification PASSED")
    else:
        print("❌ Citation metadata verification FAILED")


if __name__ == "__main__":
    main()