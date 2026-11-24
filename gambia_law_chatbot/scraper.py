"""Fetching and extracting law documents from official Gambian government sites.

Both default sites hide the ".pdf" extension behind download endpoints:
  assembly.gm : /documents/<id>/download      (list is paginated: ?page=N)
  moj.gm      : /download-file/<uuid>
so a link counts as a document if it matches any pattern in DOC_LINK_PATTERNS.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.parse
from collections import deque
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import DOWNLOAD_TIMEOUT, HTTP_HEADERS, PDF_DIR, REQUEST_TIMEOUT

LOGGER = logging.getLogger("gambia_law_chatbot.scraper")

DOC_LINK_PATTERNS = (
    re.compile(r"\.pdf$", re.I),
    re.compile(r"/documents/\d+/download/?$", re.I),
    re.compile(r"/download-file/[0-9a-f-]{8,}/?$", re.I),
)
MAX_LIST_PAGES = 60  # safety cap on pagination per source


def _clean(text: str | None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return re.sub(r"^download\s+", "", text, flags=re.I)


def _is_document_url(url: str) -> bool:
    path = urllib.parse.urlparse(url).path
    return any(p.search(path) for p in DOC_LINK_PATTERNS)


def _fetch(url: str) -> BeautifulSoup | None:
    try:
        page = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
        page.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Could not scan %s (%s)", url, exc)
        return None
    return BeautifulSoup(page.text, "html.parser")


def _next_pages(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Pagination links (?page=N) pointing at the same listing path."""
    base = urllib.parse.urlparse(page_url)
    found = []
    for anchor in soup.find_all("a", href=True):
        url = urllib.parse.urljoin(page_url, anchor["href"])
        parsed = urllib.parse.urlparse(url)
        if (parsed.netloc, parsed.path) == (base.netloc, base.path) and "page" in urllib.parse.parse_qs(parsed.query):
            found.append(url)
    return found


def find_document_links(page_url: str) -> list[tuple[str, str]]:
    """Return (title, url) for every document link on a listing, following pagination."""
    results: dict[str, str] = {}
    queue, seen = deque([page_url]), {page_url}
    while queue and len(seen) <= MAX_LIST_PAGES:
        current = queue.popleft()
        soup = _fetch(current)
        if soup is None:
            continue
        for anchor in soup.find_all("a", href=True):
            url = urllib.parse.urljoin(current, anchor["href"])
            if not _is_document_url(url):
                continue
            candidate = _clean(anchor.get("title")) or _clean(anchor.get_text())
            if candidate.lower().startswith("http"):
                candidate = ""
            if candidate and not results.get(url):
                results[url] = candidate
            else:
                results.setdefault(url, "")
        for nxt in _next_pages(soup, current):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return [(title or Path(urllib.parse.urlparse(u).path).parent.name or "document", u) for u, title in results.items()]


def download_pdf(url: str, destination: Path) -> bool:
    """Download a PDF to a destination; return True on success."""
    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Download failed for %s (%s)", url, exc)
        return False
    if b"%PDF" not in response.content[:1024]:
        LOGGER.warning("Not a PDF (skipped): %s", url)
        return False
    destination.write_bytes(response.content)
    return True


def pdf_to_text(path: Path) -> str:
    """Extract the full text of a PDF file."""
    from pypdf import PdfReader  # imported lazily; only needed for PDFs

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def collect_documents(source_urls: list[str]) -> dict[str, str]:
    """Map every document URL found on the source pages to its title."""
    documents: dict[str, str] = {}
    for source_url in source_urls:
        print(f"Scanning {source_url} ...")
        found = find_document_links(source_url)
        print(f"  {len(found)} document link(s)")
        for title, url in found:
            documents.setdefault(url, title)
    return documents


def safe_filename(title: str, url: str) -> str:
    """Turn a document title into a safe, collision-free file name."""
    slug = re.sub(r"\W+", "_", title)[:50]
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}.pdf"


def ensure_pdf_dir() -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return PDF_DIR
