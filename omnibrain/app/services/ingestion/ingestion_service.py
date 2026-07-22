import os
import time
from pathlib import Path
from typing import Any, Dict, List

from omnibrain.app.core.logging import logger, time_execution
from omnibrain.app.schemas.ingestion import ExtractedImage, IngestionResponse, TextChunk

# 1. Connect Charan's PDF Text Extractor
try:
    from scripts.prototypes.pdf_text_proto import extract_text_and_chunk
except ImportError:
    try:
        from omnibrain.app.services.ingestion.text_parser import extract_text_and_chunk
    except ImportError:

        def extract_text_and_chunk(pdf_path: str) -> List[Dict[str, Any]]:
            logger.warning("Using fallback text parsing interface")
            return []


# 2. Connect Om's Vision Extractor Engine
try:
    from omnibrain.app.services.vision.extractor import (
        extract_images as extract_images_from_pdf,
    )
except ImportError:
    try:
        from omnibrain.app.services.vision.extractor import extract_images_from_pdf
    except ImportError:

        def extract_images_from_pdf(
            pdf_path: str, output_dir: str = "output/images"
        ) -> List[Dict[str, Any]]:
            logger.warning("Using fallback vision extraction interface")
            return []


# 3. Connect Meerja's Vector Store Indexers
try:
    from omnibrain.vectorstore.indexers.text_indexer import index_text_chunks
except ImportError:

    def index_text_chunks(chunks: List[Any]) -> bool:
        logger.warning("Using fallback text vector store indexer")
        return True


try:
    from omnibrain.vectorstore.indexers.image_indexer import index_image_vectors
except ImportError:

    def index_image_vectors(images: List[Any]) -> bool:
        logger.warning("Using fallback image vector store indexer")
        return True


class IngestionService:
    """Centralized Pipeline Orchestrator for Multi-Modal PDF Ingestion.

    Coordinates Text Parsing -> Vision Extraction -> Embedding Generation -> Vector
    DB Storage.
    """

    def __init__(self):
        logger.info("Initializing IngestionService Pipeline Orchestrator")

    @time_execution("Full Ingestion Pipeline")
    def process_pdf(
        self, file_path: str, source_filename: str = None
    ) -> IngestionResponse:
        """Executes the end-to-end ingestion flow for a given PDF document."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source PDF file not found at path: {file_path}")

        filename = source_filename or Path(file_path).name
        logger.info(f"Starting ingestion workflow for file: {filename}")

        # 1. Execute Text Parsing & Chunking
        raw_chunks = self._run_text_extraction(file_path, filename)
        validated_text_chunks = self._validate_text_chunks(raw_chunks, filename)

        # 2. Execute Vision Extraction & Embeddings
        raw_images = self._run_vision_extraction(file_path, filename)
        validated_images = self._validate_images(raw_images, filename)

        # 3. Persist Multi-Modal Vector Streams into Qdrant
        self._persist_to_vector_store(validated_text_chunks, validated_images)

        # 4. Construct Final IngestionResponse metric summary
        total_pages = max(
            [c.page_number for c in validated_text_chunks]
            + [img.page_number for img in validated_images]
            + [1]
        )

        response = IngestionResponse(
            task_id=f"task_{int(time.time())}",
            status="completed",
            message="PDF multi-modal pipeline successfully executed and indexed.",
            pages_parsed=total_pages,
            text_chunks=len(validated_text_chunks),
            images_extracted=len(validated_images),
        )

        logger.info(
            f"Ingestion completed for '{filename}': "
            f"{len(validated_text_chunks)} text chunks,"
            f" {len(validated_images)} images stored."
        )
        return response

    @time_execution("Text Extraction Stage")
    def _run_text_extraction(self, file_path: str, filename: str) -> List[Any]:
        try:
            return extract_text_and_chunk(file_path)
        except Exception as e:
            logger.error(f"Error during text extraction: {str(e)}")
            return []

    @time_execution("Vision Extraction Stage")
    def _run_vision_extraction(self, file_path: str, filename: str) -> List[Any]:
        try:
            return extract_images_from_pdf(file_path, output_dir="output/images")
        except Exception as e:
            logger.error(f"Error during vision extraction: {str(e)}")
            return []

    def _validate_text_chunks(
        self, raw_chunks: List[Any], filename: str
    ) -> List[TextChunk]:
        """Converts raw dictionaries/objects
        safely into typified
        TextChunk Pydantic models."""
        validated = []
        for idx, chunk in enumerate(raw_chunks):
            try:
                if isinstance(chunk, TextChunk):
                    validated.append(chunk)
                elif isinstance(chunk, dict):
                    page_num = chunk.get("page_number", chunk.get("page", 1))
                    # Wrap the chunk_id formatting across multiple lines
                    meta = chunk.get(
                        "metadata",
                        {
                            "source": filename,
                            "section": chunk.get("section", "General"),
                            "chunk_id": chunk.get(
                                "chunk_id", f"{filename}_chunk_{idx}"
                            ),
                        },
                    )
                    validated.append(
                        TextChunk(
                            chunk_id=chunk.get("chunk_id", f"{filename}_chunk_{idx}"),
                            text=chunk.get("text", chunk.get("content", "")),
                            page_number=page_num,
                            metadata=meta,
                        )
                    )
            except Exception as err:
                logger.warning(f"Skipping malformed text chunk at index {idx}: {err}")
        return validated

    def _validate_images(
        self, raw_images: List[Any], filename: str
    ) -> List[ExtractedImage]:
        """Converts raw dictionaries/objects
        safely into typified
        ExtractedImage Pydantic models."""
        validated = []
        for idx, img in enumerate(raw_images):
            try:
                if isinstance(img, ExtractedImage):
                    validated.append(img)
                elif isinstance(img, dict):
                    page_num = img.get("page_number", img.get("page", 1))
                    validated.append(
                        ExtractedImage(
                            page_number=page_num,
                            image_path=img.get("image_path", ""),
                            dimensions=tuple(img.get("dimensions", (0, 0))),
                            caption=img.get("caption", None),
                            image_bytes=img.get("image_bytes", None),
                        )
                    )
            except Exception as err:
                logger.warning(f"Skipping malformed image object at index {idx}: {err}")
        return validated

    @time_execution("Vector Store Indexing Stage")
    def _persist_to_vector_store(
        self, text_chunks: List[TextChunk], images: List[ExtractedImage]
    ) -> None:
        if text_chunks:
            index_text_chunks(text_chunks)
        if images:
            index_image_vectors(images)
