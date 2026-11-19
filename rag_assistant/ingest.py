import hashlib

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import (
    PDF_DIR,
    DB_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    MIN_CHUNK_CHARS,
)

from rag_assistant.embeddings import get_embedding_model
from rag_assistant.pdf_loader import load_pdf_with_ocr


def _stable_id(source: str, page: int, chunk_index: int, content: str) -> str:
    """ID deterministico para un chunk: nuevo solo si cambia el contenido."""
    raw = f"{source}|{page}|{chunk_index}|{content[:500]}"
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def ingest_documents():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No se encontraron documentos PDF.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(DB_DIR)
    )

    total_files = 0
    total_chunks = 0

    for pdf_file in pdf_files:
        docs = load_pdf_with_ocr(pdf_file)
        if not docs:
            continue

        chunks = splitter.split_documents(docs)
        # Descartar chunks con texto muy corto (páginas vacías o sin contenido)
        chunks = [
            chunk for chunk in chunks
            if len((chunk.page_content or "").strip()) >= MIN_CHUNK_CHARS
        ]
        if not chunks:
            print(f"{pdf_file.name}: sin contenido útil, omitido.")
            continue

        source = str(pdf_file)

        # Borrar solo lo que tenía este archivo antes, no toda la base
        previous = vector_store.get(where={"source": source})
        if previous["ids"]:
            vector_store.delete(ids=previous["ids"])

        ids = []
        for chunk in chunks:
            chunk.metadata["source"] = source
            chunk.metadata["file_name"] = pdf_file.name
            ids.append(
                _stable_id(
                    source,
                    int(chunk.metadata.get("page", 0)),
                    len(ids),
                    chunk.page_content
                )
            )

        # Insertar en lotes para no cargar todo en memoria
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            vector_store.add_documents(
                chunks[i:i + EMBED_BATCH_SIZE],
                ids=ids[i:i + EMBED_BATCH_SIZE]
            )

        total_files += 1
        total_chunks += len(chunks)

    print(f"Ingestado: {total_files} documento(s), {total_chunks} chunks totales.")


if __name__ == "__main__":
    ingest_documents()
