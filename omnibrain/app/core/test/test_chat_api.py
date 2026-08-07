from fastapi.testclient import TestClient

from omnibrain.app.main import app


def test_chat_endpoint_scaffold(monkeypatch):
    def fake_invoke(initial_state, config=None):
        return {
            "messages": [type("Msg", (), {"content": "mock answer"})()],
            "retrieved_text": [
                {
                    "chunk_id": "chunk-1",
                    "document": "sample.pdf",
                    "page": 1,
                    "text": "retrieved text",
                    "score": 0.9,
                    "source": "sample.pdf",
                    "modality": "text",
                    "metadata": {"source": "sample.pdf"},
                }
            ],
            "retrieved_images": [
                {
                    "image_path": "output/images/example.png",
                    "page_number": 1,
                    "caption": "figure",
                    "score": 0.8,
                    "source": "sample.pdf",
                    "modality": "image",
                    "metadata": {"source": "sample.pdf"},
                }
            ],
            "thought_process": [{"agent": "Supervisor", "action": "routed"}],
        }

    monkeypatch.setattr("omnibrain.app.api.routes.chat.graph_app.invoke", fake_invoke)

    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"message": "What is in the PDF?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "mock answer"
    assert payload["images"] == ["output/images/example.png"]
