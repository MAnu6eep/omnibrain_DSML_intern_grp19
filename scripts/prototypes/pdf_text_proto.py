import fitz
import pdfplumber

pdf_path = "data/Attention_is_all_you_need.pdf"

doc = fitz.open(pdf_path)
print("Number of pages:", len(doc))

page = doc[0]
text = page.get_text()

print("\n========== pdfplumber Output ==========\n")
with pdfplumber.open(pdf_path) as pdf:
    first_page = pdf.pages[0]
    text2 = first_page.extract_text()

with open("output_pdf/pymupdf_output.txt", "w", encoding="utf-8") as f:
    f.write(text)

with open("output_pdf/pdfplumber_output.txt", "w", encoding="utf-8") as f:
    f.write(text2)
