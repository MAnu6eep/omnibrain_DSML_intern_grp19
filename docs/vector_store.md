# OmniBrain Vector Store Setup and Recovery

## 1. Overview

OmniBrain uses Qdrant as its vector database for storing and retrieving text and image embeddings.

Qdrant runs locally through Docker during development.

## 2. Qdrant Configuration

### Service Endpoints

| Service | Endpoint |
|---|---|
| REST API | http://localhost:6333 |
| Dashboard | http://localhost:6333/dashboard |
| gRPC | localhost:6334 |

### Docker Container

The project uses a Qdrant Docker container with persistent storage.

```bash
docker ps