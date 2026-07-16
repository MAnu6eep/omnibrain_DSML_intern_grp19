from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    source_name: str
    page_number: int | None = None
    chunk_index: int | None = None


@dataclass
class ImageMetadata:
    source_name: str
    page_number: int | None = None
    image_index: int | None = None
