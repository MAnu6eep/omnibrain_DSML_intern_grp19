# Standard RAG vs Omnibrain Multimodal VLM RAG

## 1. Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with a language model. Instead of relying only on the model's internal knowledge, relevant information is retrieved from an external knowledge base and provided as context for generating the response.

Omnibrain extends the conventional RAG approach by supporting both textual and visual information. This enables the system to retrieve and reason over document text as well as images such as charts and other visual content.

---

## 2. Standard RAG

A standard RAG pipeline primarily works with textual information.

### Typical flow

1. Documents are collected.
2. Text is extracted from the documents.
3. Text is divided into smaller chunks.
4. Text chunks are converted into embeddings.
5. Embeddings are stored in a vector database.
6. A user query is converted into an embedding.
7. Relevant text chunks are retrieved.
8. Retrieved text is provided as context to a language model.
9. The language model generates the final response.

### Main strength

Standard RAG works well when the required information is present in textual form.

### Main limitation

Text-only retrieval can lose information contained in visual elements such as charts, diagrams, figures, and other images.

---

## 3. Omnibrain Multimodal VLM RAG

Omnibrain extends the RAG architecture to handle both text and visual information.

The architecture documentation describes text being broken into chunks using `RecursiveCharacterTextSplitter`, with a target chunk size of 500-1000 characters and 10% overlap. Text chunks are vectorized into text embeddings.

Extracted images are transformed into dense multimodal vectors using the CLIP model. Vector payloads and metadata profiles are indexed into dual Qdrant collections.

This allows the retrieval layer to represent both textual and visual information.

### Omnibrain flow

1. Documents are ingested.
2. Text is extracted and divided into chunks.
3. Text chunks are converted into text embeddings.
4. Images are extracted from documents.
5. Images are converted into multimodal embeddings using CLIP.
6. Vector payloads and metadata are indexed in Qdrant.
7. Relevant text and/or visual information can be retrieved.
8. A Vision-Language Model (VLM) can analyze visual content.
9. The retrieved information is used to support the final response.

---

## 4. Comparative Analysis

| Aspect | Standard RAG | Omnibrain Multimodal VLM RAG |
|---|---|---|
| Primary information | Text | Text + images |
| Text processing | Text chunking and embeddings | Text chunking and embeddings |
| Image processing | Usually not part of the core pipeline | Images are extracted and represented |
| Visual embeddings | Not normally used | CLIP multimodal embeddings |
| Vector database | Stores text embeddings | Qdrant stores vector payloads and metadata |
| Visual reasoning | Limited | Supported through VLM-based analysis |
| Charts and figures | May require conversion to text | Can be represented and analyzed as visual content |
| Retrieval modality | Primarily text | Text and visual information |
| Suitable for | Text-heavy documents | Multimodal documents containing text and visuals |
| Main advantage | Simple and efficient text retrieval | Preserves and retrieves information from multiple modalities |
| Main limitation | Can miss information present only in visuals | More complex pipeline and multimodal processing requirements |

---

## 5. Why Multimodal RAG is Useful for Omnibrain

Documents can contain important information that is not completely represented by text. Charts, figures, diagrams, and other visual elements may contain trends, relationships, or data that are difficult to capture through text extraction alone.

Omnibrain addresses this limitation by incorporating image processing and multimodal embeddings into the retrieval pipeline.

The use of CLIP allows extracted images to be represented as dense multimodal vectors, while text is represented using text embeddings. These representations are indexed through the vector database layer.

The VLM component can then be used when visual understanding is required.

---

## 6. Summary

Standard RAG is primarily designed for retrieving textual context and using that context to generate responses.

Omnibrain Multimodal VLM RAG extends this concept by incorporating visual information into the retrieval and reasoning pipeline. Its architecture processes text and extracted images separately, uses text embeddings and CLIP-based multimodal embeddings, and indexes the resulting information in Qdrant.

Therefore, the key distinction is that standard RAG is primarily text-centric, whereas Omnibrain is designed to support retrieval and reasoning across both textual and visual information.
