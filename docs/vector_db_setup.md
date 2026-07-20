# Qdrant Vector Database Setup

## Overview

Qdrant is the vector database used in the OmniBrain project for storing and retrieving vector embeddings efficiently. This document explains how to run the database locally and connect to it during development.

## Prerequisites

- Docker Desktop installed and running
- Docker CLI available in the terminal

## Pull the Qdrant Image

```bash
docker pull qdrant/qdrant
```

## Run the Qdrant Container

```bash
docker run -d \
-p 6333:6333 \
-p 6334:6334 \
-v qdrant_storage:/qdrant/storage \
--name qdrant \
qdrant/qdrant
```

## Service Endpoints

| Service | Endpoint |
|---------|----------|
| REST API | http://localhost:6333 |
| Dashboard | http://localhost:6333/dashboard |
| gRPC | localhost:6334 |

## Persistent Storage

The Docker volume `qdrant_storage` stores the vector database data, ensuring it persists across container restarts.

## Verify the Setup

Check that the container is running:

```bash
docker ps
```

You should see a running container named `qdrant`.

Open the dashboard in your browser:

```
http://localhost:6333/dashboard
```

## Python Connection Example

```python
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333,
)

print(client.get_collections())
```

## Notes

- Port **6333** is used for the REST API and Dashboard.
- Port **6334** is used for gRPC communication.
- Ensure Docker Desktop is running before starting the container.
