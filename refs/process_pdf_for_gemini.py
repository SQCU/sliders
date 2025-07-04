#!/usr/bin/env -S uv run --script
# ///
# dependencies = ["PyMuPDF"]
# ///
# uv: requirements: PyMuPDF, fitz
# uv: python: ">=3.9"

import pymupdf  # PyMuPDF
import os

def process_pdf_for_gemini(pdf_path, output_dir):
    """
    Extracts text from a PDF and renders each page as a PNG image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text_output_path = os.path.join(output_dir, f"{base_name}.txt")

    try:
        document = fitz.open(pdf_path)
        text_content = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text_content += page.get_text()

            # Render page to PNG
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Render at 2x resolution
            img_output_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.png")
            pix.save(img_output_path)
            print(f"Rendered page {page_num + 1} to {img_output_path}")

        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Successfully extracted text from {pdf_path} to {text_output_path}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_pdf_for_gemini.py <input_pdf_path> <output_directory>")
        sys.exit(1)
    input_pdf = sys.argv[1]
    output_directory = sys.argv[2]
    process_pdf_for_gemini(input_pdf, output_directory)
