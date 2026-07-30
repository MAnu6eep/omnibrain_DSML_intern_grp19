from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper


def test_qdrant_wrapper_uses_host_and_port(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    monkeypatch.setenv("QDRANT_PORT", "6333")

    wrapper = QdrantClientWrapper()

    assert wrapper.url == "http://qdrant:6333"
