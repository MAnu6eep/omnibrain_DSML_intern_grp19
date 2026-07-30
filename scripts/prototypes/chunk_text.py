from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read parsed PDF text
with open("output_pdf/parsing.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = text_splitter.split_text(text)

# Save chunks
with open("output_pdf/chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"========== CHUNK {i+1} ==========\n")
        f.write(chunk)
        f.write("\n\n")

print(f"Created {len(chunks)} chunks.")
