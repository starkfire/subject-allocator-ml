from fastapi import UploadFile, File
import pymupdf


def get_text_from_pdf_stream(file: UploadFile = File(...)) -> str | None:
    """
    Retrieve texts from a PDF file stream
    """
    try:
        text = ""

        with pymupdf.open(stream=file.file.read(), filetype="pdf") as pdf:
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                text += page.get_text("text")

        text = text.replace("\n", " ")

        return text
    except pymupdf.EmptyFileError:
        return None
