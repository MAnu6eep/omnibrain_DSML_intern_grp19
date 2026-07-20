import fitz
import pdfplumber

pdf_path = "data/Attention_is_all_you_need.pdf"

doc = fitz.open(pdf_path)
print("Number of pages:", len(doc))

page = doc[0]
text = page.get_text()

with pdfplumber.open(pdf_path) as pdf:
    first_page = pdf.pages[0]
    text2 = first_page.extract_text()

with open("output_pdf/pymupdf_output.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open("output_pdf/pdfplumber_output.txt", "w", encoding="utf-8") as f:
    f.write(text2)

with open("output_pdf/15_pages.txt", "w", encoding="utf-8") as f:

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        f.write("=" * 50 + "\n")
        f.write(f"Page {page_num + 1}\n")
        f.write("=" * 50 + "\n\n")
        f.write(text)
        f.write("\n\n")



print("Done! Extracted", len(doc), "pages.")



with open("output_pdf/parsing.txt", "w", encoding="utf-8") as f:
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        f.write("=" * 60 + "\n")
        f.write(f"PAGE {page_num + 1}\n")
        f.write("=" * 60 + "\n\n")

        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            lines = para.split("\n")

            # First line as title if short
            if len(lines[0]) < 80:
                f.write(f"TITLE: {lines[0]}\n")

                if len(lines) > 1:
                    body = " ".join(lines[1:])
                    f.write(f"PARAGRAPH: {body}\n\n")
            else:
                body = " ".join(lines)
                f.write(f"PARAGRAPH: {body}\n\n")

doc.close()
