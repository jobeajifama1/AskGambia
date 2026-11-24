"""Downloading and indexing Gambian laws from government websites."""

from __future__ import annotations

import logging

from .config import Config, OllamaUnavailableError
from .ollama_client import OllamaClient
from .scraper import collect_documents, download_pdf, ensure_pdf_dir, pdf_to_text, safe_filename
from .store import Chunk, LawStore, chunk_text

LOGGER = logging.getLogger("gambia_law_chatbot.ingest")
SAVE_EVERY = 25


def ingest(config: Config) -> None:
    """Fetch, extract, chunk, and index law documents from government sites."""
    store = LawStore()
    pdf_dir = ensure_pdf_dir()

    documents = collect_documents(list(config.sources))  # {url: title}
    print(f"Found {len(documents)} PDF document link(s).")
    if not documents:
        print("No PDFs found. The site layout may link to detail pages; try --add with a deeper URL.")
        return

    client = OllamaClient(config)
    failed, no_text = [], []
    for position, (url, title) in enumerate(sorted(documents.items(), key=lambda kv: kv[1]), start=1):
        print(f"[{position}/{len(documents)}] {title}")
        local_path = pdf_dir / safe_filename(title, url)
        if not local_path.exists() and not download_pdf(url, local_path):
            failed.append(title)
            continue
        try:
            text = pdf_to_text(local_path).strip()
        except Exception as exc:  # corrupted or image-only PDFs
            LOGGER.warning("Could not read %s (%s)", local_path.name, exc)
            no_text.append(title)
            continue
        if not text:
            no_text.append(title)
            continue
        for index, text_chunk in enumerate(chunk_text(text)):
            if store.has_chunk(url, index):
                continue
            
            upper_title = title.upper()
            if "BILL" in upper_title or "DRAFT" in upper_title:
                doc_type = "Bill / draft — not necessarily enacted law"
            elif "POLICY" in upper_title or "STRATEGY" in upper_title:
                doc_type = "Policy / strategy document — adoption status requires verification"
            elif "REGULATION" in upper_title or "RULES" in upper_title:
                doc_type = "Regulation / subsidiary legislation — status requires verification"
            elif "ACT" in upper_title or "CODE" in upper_title or "CONSTITUTION" in upper_title:
                doc_type = "Act / legal instrument — current force and amendments require verification"
            else:
                doc_type = "Government publication — legal status requires verification"
            store.add(Chunk(source=title, source_url=url, chunk_index=index, text=text_chunk, document_type=doc_type))
    store.save()
    print(f"\nText extracted from {len(documents) - len(failed) - len(no_text)} document(s); "
          f"{len(no_text)} had no readable text (likely scanned images); {len(failed)} failed/not PDFs.")
    if no_text:
        print("No-text documents (need OCR):")
        for t in no_text[:15]:
            print("  -", t)
        if len(no_text) > 15:
            print(f"  ... and {len(no_text) - 15} more")
    print()

    pending = [c for c in store.chunks if c.embedding is None]
    print(f"Embedding {len(pending)} new chunk(s)... (progress is saved as it goes; safe to Ctrl+C and re-run)")
    embedded = 0
    try:
        for chunk in pending:
            try:
                chunk.embedding = client.embed(chunk.text)
                embedded += 1
            except OllamaUnavailableError:
                print("Embedding unavailable; keyword search will be used. Run: ollama pull nomic-embed-text")
                break
            if embedded % SAVE_EVERY == 0:
                store.save()
                print(f"  {embedded}/{len(pending)} embedded")
    finally:
        store.save()
    print(f"Done: {embedded} new chunk(s) embedded, {len(store.chunks)} chunks indexed in total.")
