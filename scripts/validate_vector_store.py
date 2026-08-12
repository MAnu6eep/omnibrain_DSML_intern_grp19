from time import perf_counter

from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"
COLLECTION = "omnibrain_text_chunks"
LATENCY_LIMIT_MS = 150.0


def main():
    print("=" * 60)
    print("OmniBrain Vector Store Validation")
    print("=" * 60)

    client = QdrantClient(QDRANT_URL)

    # ---------------------------------------------------------
    # 1. Qdrant connection
    # ---------------------------------------------------------
    print("\n[1] Qdrant connection")

    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]

    print("Collections:", collection_names)

    if COLLECTION not in collection_names:
        raise RuntimeError(f"Collection '{COLLECTION}' was not found.")

    print("PASS: Qdrant connection and collection found.")

    # ---------------------------------------------------------
    # 2. Collection health and configuration
    # ---------------------------------------------------------
    print("\n[2] Collection health")

    info = client.get_collection(COLLECTION)

    print("Status:", info.status)
    print("Optimizer:", info.optimizer_status)
    print("Points:", info.points_count)

    vector_config = info.config.params.vectors

    print("Vector size:", vector_config.size)
    print("Distance:", vector_config.distance)

    health_pass = (
    str(info.status).lower().endswith("green")
    and str(info.optimizer_status).lower().endswith("ok")
    and vector_config.size == 384
    and str(vector_config.distance).lower().endswith("cosine")
)

    if health_pass:
        print("PASS: Collection health/configuration is valid.")
    else:
        print("FAIL: Collection health/configuration check failed.")

    # ---------------------------------------------------------
    # 3. Get an existing vector for latency testing
    # ---------------------------------------------------------
    print("\n[3] Vector lookup latency")

    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=1,
        with_payload=False,
        with_vectors=True,
    )

    if not points or points[0].vector is None:
        raise RuntimeError("Could not retrieve a vector for latency testing.")

    vector = points[0].vector

    times = []

    for _ in range(10):
        start = perf_counter()

        client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=5,
            with_payload=False,
            with_vectors=False,
        )

        elapsed_ms = (perf_counter() - start) * 1000
        times.append(elapsed_ms)

    average_ms = sum(times) / len(times)
    max_ms = max(times)
    min_ms = min(times)

    print("Times (ms):")
    print([round(t, 2) for t in times])
    print(f"Average: {average_ms:.2f} ms")
    print(f"Minimum: {min_ms:.2f} ms")
    print(f"Maximum: {max_ms:.2f} ms")
    print(f"Requirement: < {LATENCY_LIMIT_MS:.0f} ms")

    latency_pass = max_ms < LATENCY_LIMIT_MS

    if latency_pass:
        print("PASS: Vector lookup latency is below 150 ms.")
    else:
        print("FAIL: Vector lookup latency exceeded 150 ms.")

    # ---------------------------------------------------------
    # 4. Discover document IDs
    # ---------------------------------------------------------
    print("\n[4] Document isolation")

    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    document_ids = {}

    for point in points:
        document_id = point.payload.get("document_id")

        if document_id:
            document_ids.setdefault(document_id, 0)
            document_ids[document_id] += 1

    print("Documents discovered:")

    for document_id, count in document_ids.items():
        print(f"  {document_id}: {count} points")

    if len(document_ids) < 2:
        print("WARNING: Fewer than two document IDs were found.")
        isolation_pass = False
    else:
        isolation_pass = True

        # Test two different documents.
        for document_id in list(document_ids.keys())[:2]:

            result, _ = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )

            returned_ids = {
                point.payload.get("document_id")
                for point in result
            }

            print(f"\nRequested document: {document_id}")
            print(f"Returned points: {len(result)}")
            print(f"Returned document IDs: {returned_ids}")

            if returned_ids == {document_id}:
                print("ISOLATION PASS")
            else:
                print("ISOLATION FAIL")
                isolation_pass = False

    # ---------------------------------------------------------
    # 5. Check for missing document IDs
    # ---------------------------------------------------------
    print("\n[5] Metadata quality")

    missing_document_ids = [
        point.id
        for point in points
        if not point.payload.get("document_id")
    ]

    print(
        "Points without document_id in sampled records:",
        len(missing_document_ids),
    )

    if missing_document_ids:
        print(
            "WARNING: Some legacy/sample points do not contain document_id."
        )
    else:
        print("PASS: All sampled points contain document_id.")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    print("Collection health:", "PASS" if health_pass else "FAIL")
    print("Latency (<150 ms):", "PASS" if latency_pass else "FAIL")
    print("Document isolation:", "PASS" if isolation_pass else "FAIL")

    overall_pass = health_pass and latency_pass and isolation_pass

    print("\nOverall:", "PASS" if overall_pass else "FAIL")

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()