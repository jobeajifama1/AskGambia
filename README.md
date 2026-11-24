# Gambia Legal Chatbook

A local, source-grounded AI chat interface for understanding the laws, Acts, regulations, constitutional provisions and public policies of The Gambia. It uses Ollama for private local inference and Streamlit for the browser-based chatbook.

## What changed in version 2

- Browser-based chat interface with conversation history and suggested questions.
- Stronger legal-answer rules: section-level citations where present, legal-status distinctions, no invented sections or penalties, explicit uncertainty, and practical next steps.
- Hybrid retrieval combining semantic embeddings and keyword matching; irrelevant passages are filtered rather than always returning arbitrary top results.
- Indexed passages retain source URLs and document type/status hints.
- Better chunking and safe index writes.
- Expanded starting sources: National Assembly Acts & Laws and legislative documents, and Ministry of Justice publications/downloads.
- Retains the command-line interface for technical users.

## Important limitation

This is a research prototype, not an authoritative consolidated-law database. The National Assembly document listing can include Bills as well as Acts, and a document upload/publication date is not necessarily an enactment or commencement date. The app labels Bills/drafts separately when identifiable from their titles, but users must verify amendments, commencement, repeal and current legal force against the Gazette and competent authorities. Scanned PDFs may not be text searchable; OCR is not yet included. Case-law and policies are not comprehensively indexed by default.

## Official and reference sources

1. National Assembly — enacted Acts & Laws: https://assembly.gm/documents/type/acts
2. National Assembly — wider legislative documents (including other document types): https://assembly.gm/documents
3. Ministry of Justice — Constitution, Acts, publications and Gazette downloads: https://www.moj.gm/downloads
4. Judiciary — Constitution of The Gambia PDF: https://judiciary.gov.gm/sites/default/files/2021-07/Constitution%20of%20The%20Gambia.pdf
5. Law Hub Gambia — independent legal resource with case law, legislation and secondary materials: https://www.lawhubgambia.com/

Use primary government sources for legal text. Law Hub is a useful discovery/reference source, not a substitute for checking the official text and Gazette.

## Requirements

- Python 3.10+
- Ollama: https://ollama.com/download
- Suggested local models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

You can use another Ollama chat model with `OLLAMA_MODEL` or the `--model` CLI option.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Index government documents

Start Ollama, then run:

```bash
python gambia_law_chatbot.py --ingest
```

This downloads PDF documents discovered on the configured National Assembly and Ministry of Justice pages, extracts searchable text, and stores it under `law_store/`. Re-run periodically to check for new material. Add another source page with:

```bash
python gambia_law_chatbot.py --ingest --add https://example.gov.gm/documents
```

Only add trusted sources. Review source provenance and document status before relying on indexed content.

## Start the chatbook

```bash
streamlit run app.py
```

The browser UI normally opens at `http://localhost:8501`.

## Command-line mode

```bash
python gambia_law_chatbot.py
python gambia_law_chatbot.py "Explain the Public Procurement Act"
```

## Roadmap for a public-grade service

- Gazette monitoring with version history and amendment/repeal relationships.
- OCR for scanned Acts and regulations, with page-level references.
- Structured metadata: Act number, assent date, commencement date, amendment history, repeal status and jurisdiction.
- Separate collections for Acts, subsidiary legislation, Bills, policies, Gazette notices and judgments.
- Case-law retrieval with court, date, citation and treatment/appeal status.
- Human legal editorial review and a correction/reporting workflow.
- Multilingual explanations (including Mandinka, Wolof, Fula and Jola) with clear warnings that translations are explanatory, not authoritative.
- Deployment hardening, access controls, logging policy, backups and privacy review before hosting publicly.

## Legal notice

The Chatbook provides general legal information and educational explanations, not legal advice. It can be incomplete or outdated. For advice on a particular situation, consult a qualified legal practitioner in The Gambia.

## Adding documents through the web interface

The sidebar includes a PDF uploader. Uploaded material is tagged as user-supplied and unverified; it must not be presented as official law merely because it has been indexed. The uploader currently supports text-based PDFs only.
