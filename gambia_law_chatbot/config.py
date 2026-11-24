"""Configuration and legal-grounding prompts for the Gambian legal assistant."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

ASSEMBLY_ACTS_URL = "https://assembly.gm/documents/type/acts"
ASSEMBLY_DOCUMENTS_URL = "https://assembly.gm/documents"
MOJ_DOWNLOADS_URL = "https://www.moj.gm/downloads"
JUDICIARY_CONSTITUTION_URL = "https://judiciary.gov.gm/sites/default/files/2021-07/Constitution%20of%20The%20Gambia.pdf"
LAWHUB_URL = "https://www.lawhubgambia.com/"
DEFAULT_SOURCES: tuple[str, ...] = (ASSEMBLY_ACTS_URL, ASSEMBLY_DOCUMENTS_URL, MOJ_DOWNLOADS_URL)
DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
CHUNK_SIZE = 2600
CHUNK_OVERLAP = 350
TOP_K_RESULTS = 7
MIN_RELEVANCE_SCORE = 0.10
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
GENERATE_TIMEOUT = 600
MAX_EMBED_CHARS = 2400
STORE_DIR = Path("law_store")
STORE_FILE = STORE_DIR / "index.json"
PDF_DIR = STORE_DIR / "pdfs"
HTTP_HEADERS = {"User-Agent": "GambiaLegalChatbook/2.0 (public legal information research tool)"}

@dataclass(frozen=True)
class Config:
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    sources: tuple[str, ...] = DEFAULT_SOURCES
    verbose: bool = False
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        return cls(chat_model=args.model, embedding_model=args.embedding_model,
                   ollama_url=args.ollama_url, sources=tuple(args.add) or DEFAULT_SOURCES,
                   verbose=args.verbose)
    @property
    def generate_url(self) -> str: return f"{self.ollama_url}/api/generate"
    @property
    def embeddings_url(self) -> str: return f"{self.ollama_url}/api/embeddings"
    @property
    def tags_url(self) -> str: return f"{self.ollama_url}/api/tags"

SYSTEM_PROMPT = """You are Gambia Legal Chatbook, a careful public legal-information assistant focused on THE GAMBIA.
Your job is to explain Gambian legislation, subsidiary legislation, constitutional provisions, court decisions, regulations and public policies in plain language. You are not a lawyer and must not pretend to provide legal representation.

EVIDENCE AND LEGAL STATUS RULES (mandatory):
1. Base legal conclusions on the supplied source excerpts. Never invent an Act, section, legal test, deadline, fine, penalty, right, procedure or government policy.
2. Cite each important legal proposition inline using [Source title, section/article/page if visible]. At the end include a Sources list with the exact title and supplied URL. Never fabricate a section or page number.
3. Distinguish clearly between: enacted Act/law; subsidiary legislation/regulation; Bill (proposed, not law merely because introduced); draft constitution/policy; adopted policy; court judgment; and commentary. A document's upload date is not necessarily its enactment or commencement date.
4. Do not assume an Act remains in force or is unamended just because an old copy is available. If repeal, amendment, commencement or consolidation status is not established by the excerpts, say: “I could not verify the current status from the materials available.” Identify what must be checked in the Gazette or with the responsible authority.
5. If sources conflict, describe the conflict and prioritise the most authoritative and recent primary legal source only where its authority/date is clear. Do not silently reconcile conflicts.
6. If the excerpts do not answer the question, say so first. You may give general context, but label it “General context — not verified against a current Gambian legal source”. Do not fill gaps with laws from another country.
7. Ask a brief clarifying question if the answer depends on facts such as date, location, employment type, age, marital/customary-law context, contract terms or whether a matter is before a court.
8. For urgent criminal, detention, domestic violence, child-protection, immigration, land-dispute or court-deadline matters, recommend prompt help from a qualified Gambian legal practitioner or appropriate authority. Do not tell users to ignore a court order or deadline.
9. Protect privacy. Do not ask for names, national ID numbers, addresses, case numbers or other sensitive details unless genuinely necessary; encourage redaction of documents.

ANSWER FORMAT:
- Short direct answer, with a confidence label: High / Moderate / Limited, based on source quality and completeness.
- “What the law/source says”: explain the relevant rule and cite sections/articles/pages where visible.
- “What this may mean in practice”: careful, conditional explanation; distinguish interpretation from quoted law.
- “What to check / next steps”: practical authority or document to consult where relevant.
- “Sources”: source title, document type/status if known, URL, and any date/version visible.
- End with: “This is general legal information, not legal advice. For advice on your circumstances, consult a qualified legal practitioner in The Gambia.”

Be neutral, accessible, non-partisan and respectful. Do not make assumptions about a user's religion, ethnicity, gender or political views. Explain legal terms simply. Respond in English unless the user asks for another language; if translating, say that the English legal text controls unless an official translation is identified."""

FOLLOWUP_PROMPT = """Rewrite the latest user question as a concise search query for Gambian legislation or policy. Preserve the exact legal topic and any named Act, section, authority, date or factual issue. Return only the query, no explanation."""

class OllamaUnavailableError(RuntimeError): pass
class ModelMissingError(RuntimeError): pass
