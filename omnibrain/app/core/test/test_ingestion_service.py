import pytest
from pathlib import Path
from omnibrain.app.services.ingestion.ingestion_service import IngestionService

def test_ingestion_service_initialization():
    service = IngestionService()
    assert service is not None

def test_ingestion_service_nonexistent_file():
    service = IngestionService()
    with pytest.raises(FileNotFoundError):
        service.process_pdf("non_existent_file.pdf")