from omnibrain.vectorstore.retrievers.text_retriever import search_text_chunks


def test_text_retrieval_filters_empty_and_duplicate_results(monkeypatch):
    class FakeEmbeddingModel:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3]]

    class FakePoint:
        def __init__(self, point_id, payload, score):
            self.id = point_id
            self.payload = payload
            self.score = score

    class FakeResponse:
        def __init__(self):
            self.points = [
                FakePoint(
                    "1",
                    {
                        "chunk_id": "a",
                        "text": "first",
                        "document": "doc.pdf",
                        "page_number": 1,
                        "source": "doc.pdf",
                        "modality": "text",
                    },
                    0.9,
                ),
                FakePoint(
                    "2",
                    {
                        "chunk_id": "a",
                        "text": "duplicate",
                        "document": "doc.pdf",
                        "page_number": 1,
                        "source": "doc.pdf",
                        "modality": "text",
                    },
                    0.8,
                ),
                FakePoint(
                    "3",
                    {
                        "chunk_id": "b",
                        "text": "",
                        "document": "doc.pdf",
                        "page_number": 2,
                        "source": "doc.pdf",
                        "modality": "text",
                    },
                    0.7,
                ),
            ]

    class FakeClient:
        def query_points(self, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "omnibrain.vectorstore.retrievers.text_retriever.get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "omnibrain.vectorstore.retrievers.text_retriever.get_client",
        lambda: FakeClient(),
    )

    results = search_text_chunks("hello world")

    assert len(results) == 1
    assert results[0]["text"] == "first"
