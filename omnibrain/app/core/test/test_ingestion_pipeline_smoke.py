from omnibrain.app.services.ingestion.ingestion_service import IngestionService


def test_ingestion_pipeline_smoke(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% smoke test\n")

    monkeypatch.setattr(
        "omnibrain.app.services.ingestion.ingestion_service.extract_text_and_chunk",
        lambda file_path: [
            {
                "chunk_id": "sample_p1_c0",
                "text": "hello world",
                "page_number": 1,
                "source": "sample.pdf",
                "source_path": str(file_path),
                "metadata": {"source": "sample.pdf", "modality": "text"},
            }
        ],
    )
    monkeypatch.setattr(
        "omnibrain.app.services.ingestion.ingestion_service.extract_images_from_pdf",
        lambda file_path, output_dir="output/images": [
            {
                "page_number": 1,
                "image_path": str(tmp_path / "figure.png"),
                "dimensions": (640, 480),
                "caption": "figure caption",
                "image_bytes": b"png",
            }
        ],
    )
    monkeypatch.setattr(
        "omnibrain.app.services.ingestion.ingestion_service.index_text_chunks",
        lambda chunks: True,
    )
    monkeypatch.setattr(
        "omnibrain.app.services.ingestion.ingestion_service.index_image_vectors",
        lambda images: True,
    )

    service = IngestionService()
    result = service.process_pdf(str(pdf_path), source_filename=pdf_path.name)

    assert result.status in {"completed", "partial"}
    assert result.text_chunks == 1
    assert result.images_extracted == 1
