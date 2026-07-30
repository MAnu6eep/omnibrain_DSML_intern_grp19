import os
import time
from pathlib import Path
from typing import Any, Dict, List

from omnibrain.app.core.logging import logger, time_execution
from omnibrain.app.schemas.ingestion import ExtractedImage, IngestionResponse, TextChunk

# 1. Connect Charan's PDF Text Extractor
try:
    from omnibrain.app.services.ingestion.text_parser import extract_text_and_chunk
except ImportError:
    try:
        from scripts.prototypes.pdf_text_proto import extract_text_and_chunk
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
        return self._process_single_pdf(file_path, source_filename=source_filename)

    def process_path(self, source_path: str) -> IngestionResponse:
        """Process a single PDF file or a folder containing PDFs."""

        source = Path(source_path)
        if source.is_dir():
            pdf_files = sorted(path for path in source.rglob("*.pdf") if path.is_file())

            if not pdf_files:
                raise FileNotFoundError(
                    f"No PDF files found under directory: {source_path}"
                )

            aggregate_pages = 0
            aggregate_chunks = 0
            aggregate_images = 0
            warnings: List[str] = []

            for pdf_file in pdf_files:
                result = self._process_single_pdf(str(pdf_file), pdf_file.name)
                aggregate_pages += result.pages_parsed
                aggregate_chunks += result.text_chunks
                aggregate_images += result.images_extracted
                warnings.extend(result.warnings)

            status = "completed" if not warnings else "partial"

            return IngestionResponse(
                task_id=f"task_{int(time.time())}",
                status=status,
                message="Folder ingestion completed for the available PDF files.",
                pages_parsed=aggregate_pages,
                text_chunks=aggregate_chunks,
                images_extracted=aggregate_images,
                warnings=warnings,
            )

        return self._process_single_pdf(source_path, source_filename=source.name)

    def _process_single_pdf(
        self, file_path: str, source_filename: str = None
    ) -> IngestionResponse:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source PDF file not found at path: {file_path}")

        filename = source_filename or Path(file_path).name
        logger.info("Starting ingestion workflow for file: %s", filename)

        warnings: List[str] = []

        raw_chunks = self._run_text_extraction(file_path, filename)
        validated_text_chunks = self._validate_text_chunks(raw_chunks, filename)
        if not validated_text_chunks:
            warnings.append(
                "No text chunks were produced. Table extraction remains partial."
            )

        raw_images = self._run_vision_extraction(file_path, filename)
        validated_images = self._validate_images(raw_images, filename)
        if not validated_images:
            warnings.append("No embedded images were extracted from the document.")

        persist_warning = self._persist_to_vector_store(
            validated_text_chunks, validated_images
        )
        if persist_warning:
            warnings.append(persist_warning)

        total_pages = max(
            [c.page_number for c in validated_text_chunks]
            + [img.page_number for img in validated_images]
            + [1]
        )

        status = "completed" if not warnings else "partial"

        response = IngestionResponse(
            task_id=f"task_{int(time.time())}",
            status=status,
            message=(
                "PDF multi-modal pipeline successfully executed and indexed."
                if status == "completed"
                else "PDF multi-modal pipeline completed with warnings."
            ),
            pages_parsed=total_pages,
            text_chunks=len(validated_text_chunks),
            images_extracted=len(validated_images),
            warnings=warnings,
        )

        logger.info(
            "Ingestion completed for '%s': %s text chunks, %s images stored.",
            filename,
            len(validated_text_chunks),
            len(validated_images),
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
                    if chunk.text.strip():
                        validated.append(chunk)
                elif isinstance(chunk, dict):
                    page_num = chunk.get("page_number", chunk.get("page", 1))
                    text = (chunk.get("text", chunk.get("content", "")) or "").strip()
                    if not text:
                        continue

                    meta = chunk.get("metadata", {}) or {}
                    meta.setdefault("source", chunk.get("source", filename))
                    meta.setdefault("source_path", chunk.get("source_path", ""))
                    meta.setdefault("section", chunk.get("section", "General"))
                    meta.setdefault(
                        "chunk_id", chunk.get("chunk_id", f"{filename}_chunk_{idx}")
                    )
                    meta.setdefault("modality", "text")
                    validated.append(
                        TextChunk(
                            chunk_id=chunk.get("chunk_id", f"{filename}_chunk_{idx}"),
                            text=text,
                            page_number=page_num,
                            source=chunk.get("source", filename),
                            source_path=chunk.get("source_path", ""),
                            modality=chunk.get("modality", "text"),
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
                    if img.image_path:
                        validated.append(img)
                elif isinstance(img, dict):
                    page_num = img.get("page_number", img.get("page", 1))
                    image_path = img.get("image_path", "")
                    if not image_path:
                        continue
                    validated.append(
                        ExtractedImage(
                            page_number=page_num,
                            image_path=image_path,
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
    ) -> str:
        warnings: List[str] = []

        try:
            if text_chunks:
                index_text_chunks(text_chunks)
        except Exception as exc:
            logger.error("Text vector store indexing failed: %s", exc)
            warnings.append(f"Text vector store indexing failed: {exc}")

        try:
            if images:
                index_image_vectors(images)
        except Exception as exc:
            logger.error("Image vector store indexing failed: %s", exc)
            warnings.append(f"Image vector store indexing failed: {exc}")

        return "; ".join(warnings)
