import pytest
from fastapi.testclient import TestClient
from omnibrain.app.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200

def test_upload_invalid_file_type(client):
    # Tests that non-PDF uploads are rejected properly
    files = {"file": ("test.txt", b"Hello World", "text/plain")}
    response = client.post("/api/v1/ingestion/upload", files=files)
    # Expects 400 Bad Request or 422 Unprocessable Entity
    assert response.status_code in [400, 422]