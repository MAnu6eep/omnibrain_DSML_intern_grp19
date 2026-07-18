import fitz  # PyMuPDF

# Path to the sample PDF
pdf_path = "data/sample2.pdf"

try:
    # Open the PDF
    pdf = fitz.open(pdf_path)

    print("=" * 50)
    print("PDF opened successfully!")
    print(f"File: {pdf_path}")
    print(f"Total Pages: {len(pdf)}")
    print("=" * 50)

    # Loop through every page
    for page_number in range(len(pdf)):

        page = pdf.load_page(page_number)

        images = page.get_images(full=True)

        print(f"\nPage {page_number + 1}")

        if len(images) == 0:
            print("No images found.")

        else:
            print(f"Total Images: {len(images)}")

            for index, image in enumerate(images, start=1):

                xref = image[0]

                print(f"  Image {index}")
                print(f"     XREF ID : {xref}")

                # Extract image
                base_image = pdf.extract_image(xref)

                image_bytes = base_image["image"]

                image_extension = base_image["ext"]

                image_name = f"page_{page_number + 1}_image_{index}.{image_extension}"

                with open(f"output/images/{image_name}", "wb") as image_file:
                    image_file.write(image_bytes)

                print(f"     Saved as: {image_name}")

    pdf.close()

except Exception as e:
    print("Error opening PDF:")
    print(e)
