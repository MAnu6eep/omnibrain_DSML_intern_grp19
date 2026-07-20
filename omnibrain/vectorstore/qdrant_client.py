from dataclasses import dataclass

from qdrant_client import QdrantClient

from omnibrain.app.core.config import settings


@dataclass
class QdrantClientWrapper:
    url: str = settings.qdrant_url
    api_key: str = settings.qdrant_api_key

    def client(self) -> QdrantClient:
        kwargs = {"url": self.url}

        if self.api_key:
            kwargs["api_key"] = self.api_key

        return QdrantClient(**kwargs)