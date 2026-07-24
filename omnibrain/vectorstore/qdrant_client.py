import os
from dataclasses import dataclass

from qdrant_client import QdrantClient

from omnibrain.app.core.config import settings


@dataclass
class QdrantClientWrapper:
    url: str = None
    api_key: str = None

    def __post_init__(self):
        # Allow environment override or default to settings
        # Split the os.getenv assignment
        if self.url is None:
            self.url = os.getenv("QDRANT_URL") or getattr(
                settings, "qdrant_url", "http://localhost:6333"
            )

        # Replace service name with localhost if running outside Docker
        if "omnibrain_vector_db" in self.url or "qdrant" in self.url:
            if not os.path.exists("/.dockerenv"):
                self.url = "http://localhost:6333"

        if self.api_key is None:
            self.api_key = os.getenv("QDRANT_API_KEY") or getattr(
                settings, "qdrant_api_key", None
            )

    def client(self) -> QdrantClient:
        kwargs = {"url": self.url}

        if self.api_key:
            kwargs["api_key"] = self.api_key

        return QdrantClient(**kwargs)
