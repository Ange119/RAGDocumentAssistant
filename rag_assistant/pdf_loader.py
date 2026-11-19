"""Carga de PDFs: texto real primero, OCR (PyMuPDF + Tesseract) como fallback.

Los PDFs escaneados (sin capa de texto legible)
así que las páginas sin texto legible se renderizan a imagen y se leen con OCR.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag_assistant.config import MIN_TEXT_CHARS, OCR_DPI, OCR_LANG, TESSERACT_CMD


def _ocr_page_text(page, dpi: int) -> str:
    """Renderiza una página del PDF a imagen y lee el texto con Tesseract."""
    import pymupdf
    import pytesseract

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    pixmap = page.get_pixmap(dpi=dpi)
    image = pixmap.pil_image().convert("L")

    return pytesseract.image_to_string(image, lang=OCR_LANG).strip()


def _has_text(doc: Document) -> bool:
    text = (doc.page_content or "").strip()
    return len(text) >= MIN_TEXT_CHARS


def load_pdf_with_ocr(path) -> list[Document]:
    """Carga un PDF, usando OCR en las páginas sin texto legible."""
    path = Path(path)

    try:
        docs = PyPDFLoader(str(path)).load()
    except Exception as exc:
        print(f"Advertencia: no se pudo cargar {path.name}: {exc}")
        return []

    if not docs:
        print(f"{path.name}: no contiene páginas leíbles.")
        return []

    text_pages = sum(1 for doc in docs if _has_text(doc))
    needs_ocr = [doc for doc in docs if not _has_text(doc)]

    if not needs_ocr:
        for doc in docs:
            doc.metadata["extraction"] = "text"
        print(
            f"{path.name}: {len(docs)} página(s), todas con texto legible."
        )
        return docs

    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError(
            "Para OCR se necesita 'pymupdf', 'pytesseract' y 'pillow': "
            "pip install pymupdf pytesseract pillow"
        ) from exc

    pdf_doc = pymupdf.open(str(path))
    ocred = 0
    try:
        for doc in needs_ocr:
            page_index = int(doc.metadata.get("page", 0))
            if not (0 <= page_index < len(pdf_doc)):
                continue
            try:
                text = _ocr_page_text(pdf_doc[page_index], dpi=OCR_DPI)
            except Exception as exc:
                print(
                    f"  OCR falló en {path.name} pág. {page_index + 1}: {exc}"
                )
                continue
            if text:
                doc.page_content = text
            ocred += 1
    finally:
        pdf_doc.close()

    remaining_empty = [
        doc.metadata.get("page", 0) + 1
        for doc in docs
        if not _has_text(doc)
    ]

    for doc in docs:
        doc.metadata["extraction"] = "text" if _has_text(doc) else "none"

    print(
        f"{path.name}: {len(docs)} página(s), {text_pages} con texto, "
        f"{ocred} con OCR."
    )
    if remaining_empty:
        print(
            f"  -> páginas sin texto obtenible: {remaining_empty}"
        )

    return docs
