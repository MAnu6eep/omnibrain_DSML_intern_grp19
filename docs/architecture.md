# OmniBrain Ingestion Pipeline Architecture

This document maps the sequential data flow of a PDF document through the OmniBrain system during Days 2 & 3.

## Data Flow Pipeline

1. **API Upload Gateway (Manav - Backend)**
   - Receives multi-part form data at `POST /api/v1/ingestion/upload`.
   - Validates the PDF mime-type and streams it temporarily to the disk.

2. **Parallel Processing Layer (Charan & Om - Document Parsing)**
   - **Text Extraction:** PyMuPDF streams text page-by-page, matching the `ExtractedTextPage` schema.
   - **Image Extraction:** Embedded graphics are extracted, saved to disk, and mapped to the `ExtractedImage` schema.

3. **Transformation & Embedding Layer (Charan, Om, & Meerja)**
   - Text is broken into chunks using `RecursiveCharacterTextSplitter` (Target: 500-1000 chars, 10% overlap) into `TextChunk` format.
   - Extracted images are transformed into dense multi-modal vectors using the CLIP model.
   - Chunks are vectorized into text embeddings.

4. **Persistence Layer (Meerja - Vector DB)**
   - Vector payloads and metadata profiles are indexed simultaneously into dual Qdrant collections.