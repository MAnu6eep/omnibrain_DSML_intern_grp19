import os
import time

from pathlib import Path
from typing import Any, Dict, List

from omnibrain.app.core.logging import (
    logger,
    time_execution,
)

from omnibrain.app.schemas.ingestion import (
    ExtractedImage,
    IngestionResponse,
    TextChunk,
)


# ============================================================
# TEXT EXTRACTION
# ============================================================

try:

    from omnibrain.app.services.ingestion.text_parser import (
        extract_text_and_chunk,
    )

except ImportError:

    try:

        from scripts.prototypes.pdf_text_proto import (
            extract_text_and_chunk,
        )

    except ImportError:

        def extract_text_and_chunk(
            pdf_path: str,
        ) -> List[Dict[str, Any]]:

            logger.warning(
                "Using fallback text parsing interface"
            )

            return []


# ============================================================
# IMAGE EXTRACTION
# ============================================================

try:

    from omnibrain.app.services.vision.extractor import (
        extract_images as extract_images_from_pdf,
    )

except ImportError:

    try:

        from omnibrain.app.services.vision.extractor import (
            extract_images_from_pdf,
        )

    except ImportError:

        def extract_images_from_pdf(
            pdf_path: str,
            output_dir: str = "output/images",
        ) -> List[Dict[str, Any]]:

            logger.warning(
                "Using fallback vision extraction interface"
            )

            return []


# ============================================================
# TEXT INDEXER
# ============================================================

try:

    from omnibrain.vectorstore.indexers.text_indexer import (
        index_text_chunks,
    )

except ImportError:

    def index_text_chunks(
        chunks: List[Any],
    ) -> bool:

        logger.warning(
            "Using fallback text vector store indexer"
        )

        return True


# ============================================================
# IMAGE INDEXER
# ============================================================

try:

    from omnibrain.vectorstore.indexers.image_indexer import (
        index_image_vectors,
    )

except ImportError:

    def index_image_vectors(
        images: List[Any],
    ) -> bool:

        logger.warning(
            "Using fallback image vector store indexer"
        )

        return True


# ============================================================
# INGESTION SERVICE
# ============================================================

class IngestionService:

    def __init__(self):

        logger.info(
            "Initializing IngestionService Pipeline Orchestrator"
        )

    # ========================================================
    # PUBLIC PDF PROCESSOR
    # ========================================================

    @time_execution(
        "Full Ingestion Pipeline"
    )
    def process_pdf(
        self,
        file_path: str,
        source_filename: str = None,
        document_id: str = None,
    ) -> IngestionResponse:

        return self._process_single_pdf(
            file_path=file_path,
            source_filename=source_filename,
            document_id=document_id,
        )

    # ========================================================
    # PROCESS PATH
    # ========================================================

    def process_path(
        self,
        source_path: str,
    ) -> IngestionResponse:

        source = Path(
            source_path
        )

        if source.is_dir():

            pdf_files = sorted(
                path
                for path in source.rglob(
                    "*.pdf"
                )
                if path.is_file()
            )

            if not pdf_files:

                raise FileNotFoundError(
                    f"No PDF files found under directory: {source_path}"
                )

            aggregate_pages = 0
            aggregate_chunks = 0
            aggregate_images = 0

            warnings = []

            for pdf_file in pdf_files:

                document_id = str(
                    __import__(
                        "uuid"
                    ).uuid4()
                )

                result = self._process_single_pdf(
                    file_path=str(
                        pdf_file
                    ),
                    source_filename=pdf_file.name,
                    document_id=document_id,
                )

                aggregate_pages += (
                    result.pages_parsed
                )

                aggregate_chunks += (
                    result.text_chunks
                )

                aggregate_images += (
                    result.images_extracted
                )

                warnings.extend(
                    result.warnings
                )

            status = (
                "completed"
                if not warnings
                else "partial"
            )

            return IngestionResponse(
                task_id=(
                    f"task_{int(time.time())}"
                ),
                status=status,
                message=(
                    "Folder ingestion completed."
                ),
                pages_parsed=aggregate_pages,
                text_chunks=aggregate_chunks,
                images_extracted=aggregate_images,
                warnings=warnings,
            )

        document_id = str(
            __import__(
                "uuid"
            ).uuid4()
        )

        return self._process_single_pdf(
            file_path=source_path,
            source_filename=source.name,
            document_id=document_id,
        )

    # ========================================================
    # SINGLE PDF
    # ========================================================

    def _process_single_pdf(
        self,
        file_path: str,
        source_filename: str = None,
        document_id: str = None,
    ) -> IngestionResponse:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Source PDF file not found at path: {file_path}"
            )

        filename = (
            source_filename
            or Path(file_path).name
        )

        logger.info(
            "Starting ingestion workflow for file: %s",
            filename,
        )

        if not document_id:

            logger.warning(
                "No document_id provided. "
                "Generating one automatically."
            )

            document_id = str(
                __import__(
                    "uuid"
                ).uuid4()
            )

        logger.info(
            "Document ID for '%s': %s",
            filename,
            document_id,
        )

        warnings = []

        # ====================================================
        # TEXT
        # ====================================================

        raw_chunks = (
            self._run_text_extraction(
                file_path,
                filename,
            )
        )

        validated_text_chunks = (
            self._validate_text_chunks(
                raw_chunks,
                filename,
                document_id,
            )
        )

        if not validated_text_chunks:

            warnings.append(
                "No text chunks were produced."
            )

        # ====================================================
        # IMAGES
        # ====================================================

        raw_images = (
            self._run_vision_extraction(
                file_path,
                filename,
            )
        )

        validated_images = (
            self._validate_images(
                raw_images,
                filename,
                document_id,
            )
        )

        if not validated_images:

            warnings.append(
                "No embedded images were extracted."
            )

        # ====================================================
        # QDRANT
        # ====================================================

        persist_warning = (
            self._persist_to_vector_store(
                validated_text_chunks,
                validated_images,
            )
        )

        if persist_warning:

            warnings.append(
                persist_warning
            )

        # ====================================================
        # PAGE COUNT
        # ====================================================

        pages = [
            chunk.page_number
            for chunk in validated_text_chunks
        ]

        pages.extend(
            image.page_number
            for image in validated_images
        )

        total_pages = max(
            pages + [1]
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        status = (
            "completed"
            if not warnings
            else "partial"
        )

        return IngestionResponse(
            task_id=(
                f"task_{int(time.time())}"
            ),
            status=status,
            message=(
                "PDF multi-modal pipeline successfully executed and indexed."
                if status == "completed"
                else
                "PDF multi-modal pipeline completed with warnings."
            ),
            pages_parsed=total_pages,
            text_chunks=len(
                validated_text_chunks
            ),
            images_extracted=len(
                validated_images
            ),
            warnings=warnings,
        )

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    @time_execution(
        "Text Extraction Stage"
    )
    def _run_text_extraction(
        self,
        file_path: str,
        filename: str,
    ) -> List[Any]:

        try:

            return extract_text_and_chunk(
                file_path
            )

        except Exception as exc:

            logger.error(
                "Text extraction failed: %s",
                exc,
            )

            return []

    # ========================================================
    # IMAGE EXTRACTION
    # ========================================================

    @time_execution(
        "Vision Extraction Stage"
    )
    def _run_vision_extraction(
        self,
        file_path: str,
        filename: str,
    ) -> List[Any]:

        try:

            return extract_images_from_pdf(
                file_path,
                output_dir="output/images",
            )

        except Exception as exc:

            logger.error(
                "Image extraction failed: %s",
                exc,
            )

            return []

    # ========================================================
    # TEXT VALIDATION
    # ========================================================

    def _validate_text_chunks(
        self,
        raw_chunks: List[Any],
        filename: str,
        document_id: str,
    ) -> List[TextChunk]:

        validated = []

        for idx, chunk in enumerate(
            raw_chunks
        ):

            try:

                if isinstance(
                    chunk,
                    TextChunk,
                ):

                    text = (
                        chunk.text
                        or ""
                    ).strip()

                    if not text:
                        continue

                    metadata = dict(
                        chunk.metadata
                        or {}
                    )

                    chunk_id = (
                        chunk.chunk_id
                        or f"{filename}_chunk_{idx}"
                    )

                    source = (
                        chunk.source
                        or filename
                    )

                    source_path = (
                        chunk.source_path
                        or ""
                    )

                    metadata[
                        "document_id"
                    ] = document_id

                    metadata[
                        "chunk_id"
                    ] = chunk_id

                    metadata[
                        "page_number"
                    ] = chunk.page_number

                    metadata[
                        "source"
                    ] = source

                    metadata[
                        "source_path"
                    ] = source_path

                    metadata[
                        "modality"
                    ] = (
                        chunk.modality
                        or "text"
                    )

                    validated.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            page_number=chunk.page_number,
                            text=text,
                            source=source,
                            source_path=source_path,
                            modality="text",
                            metadata=metadata,
                        )
                    )

                elif isinstance(
                    chunk,
                    dict,
                ):

                    text = (
                        chunk.get(
                            "text",
                            chunk.get(
                                "content",
                                "",
                            ),
                        )
                        or ""
                    ).strip()

                    if not text:
                        continue

                    page_number = chunk.get(
                        "page_number",
                        chunk.get(
                            "page",
                            1,
                        ),
                    )

                    metadata = dict(
                        chunk.get(
                            "metadata",
                            {},
                        )
                        or {}
                    )

                    chunk_id = (
                        chunk.get(
                            "chunk_id"
                        )
                        or metadata.get(
                            "chunk_id"
                        )
                        or f"{filename}_chunk_{idx}"
                    )

                    source = (
                        chunk.get(
                            "source"
                        )
                        or metadata.get(
                            "source"
                        )
                        or filename
                    )

                    source_path = (
                        chunk.get(
                            "source_path"
                        )
                        or metadata.get(
                            "source_path"
                        )
                        or ""
                    )

                    metadata[
                        "document_id"
                    ] = document_id

                    metadata[
                        "chunk_id"
                    ] = chunk_id

                    metadata[
                        "page_number"
                    ] = page_number

                    metadata[
                        "source"
                    ] = source

                    metadata[
                        "source_path"
                    ] = source_path

                    metadata[
                        "modality"
                    ] = "text"

                    validated.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            page_number=page_number,
                            text=text,
                            source=source,
                            source_path=source_path,
                            modality="text",
                            metadata=metadata,
                        )
                    )

            except Exception as exc:

                logger.warning(
                    "Skipping malformed text chunk %s: %s",
                    idx,
                    exc,
                )

        return validated

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    def _validate_images(
        self,
        raw_images: List[Any],
        filename: str,
        document_id: str,
    ) -> List[ExtractedImage]:

        validated = []

        for idx, image in enumerate(
            raw_images
        ):

            try:

                if isinstance(
                    image,
                    ExtractedImage,
                ):

                    if not image.image_path:
                        continue

                    image_id = (
                        image.image_id
                        or str(
                            __import__(
                                "uuid"
                            ).uuid4()
                        )
                    )

                    validated.append(
                        ExtractedImage(
                            image_id=image_id,
                            document_id=document_id,
                            page_number=image.page_number,
                            image_path=image.image_path,
                            source=(
                                image.source
                                or filename
                            ),
                            source_path=(
                                image.source_path
                                or ""
                            ),
                            dimensions=image.dimensions,
                            caption=image.caption,
                            image_bytes=image.image_bytes,
                            modality="image",
                        )
                    )

                elif isinstance(
                    image,
                    dict,
                ):

                    image_path = (
                        image.get(
                            "image_path",
                            "",
                        )
                        or ""
                    )

                    if not image_path:
                        continue

                    page_number = image.get(
                        "page_number",
                        image.get(
                            "page",
                            1,
                        ),
                    )

                    image_id = (
                        image.get(
                            "image_id"
                        )
                        or str(
                            __import__(
                                "uuid"
                            ).uuid4()
                        )
                    )

                    source = (
                        image.get(
                            "source"
                        )
                        or filename
                    )

                    source_path = (
                        image.get(
                            "source_path"
                        )
                        or ""
                    )

                    dimensions = tuple(
                        image.get(
                            "dimensions",
                            (0, 0),
                        )
                    )

                    validated.append(
                        ExtractedImage(
                            image_id=str(
                                image_id
                            ),
                            document_id=document_id,
                            page_number=page_number,
                            image_path=image_path,
                            source=source,
                            source_path=source_path,
                            dimensions=dimensions,
                            caption=image.get(
                                "caption"
                            ),
                            image_bytes=image.get(
                                "image_bytes"
                            ),
                            modality="image",
                        )
                    )

            except Exception as exc:

                logger.warning(
                    "Skipping malformed image %s: %s",
                    idx,
                    exc,
                )

        return validated

    # ========================================================
    # QDRANT INDEXING
    # ========================================================

    @time_execution(
        "Vector Store Indexing Stage"
    )
    def _persist_to_vector_store(
        self,
        text_chunks: List[TextChunk],
        images: List[ExtractedImage],
    ) -> str:

        warnings = []

        try:

            if text_chunks:

                index_text_chunks(
                    text_chunks
                )

        except Exception as exc:

            logger.error(
                "Text vector store indexing failed: %s",
                exc,
            )

            warnings.append(
                f"Text vector store indexing failed: {exc}"
            )

        try:

            if images:

                index_image_vectors(
                    images
                )

        except Exception as exc:

            logger.error(
                "Image vector store indexing failed: %s",
                exc,
            )

            warnings.append(
                f"Image vector store indexing failed: {exc}"
            )

        return "; ".join(
            warnings
        )