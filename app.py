"""Streamlit interface for Gambia Legal Chatbook."""
from __future__ import annotations
import os
import streamlit as st
from gambia_law_chatbot.config import Config, DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_URL
from gambia_law_chatbot.chat import ask
from gambia_law_chatbot.store import LawStore, Chunk, chunk_text
from gambia_law_chatbot.scraper import pdf_to_text
from gambia_law_chatbot.config import PDF_DIR
from gambia_law_chatbot.ollama_client import OllamaClient
from gambia_law_chatbot.ollama_client import OllamaUnavailableError

st.set_page_config(page_title="Gambia Legal Chatbook", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
:root { --ink:#172b27; --muted:#64736e; --green:#145b45; --gold:#d7a64a; }
.block-container {max-width: 1050px; padding-top: 2rem; padding-bottom: 4rem;}
.hero {background:linear-gradient(120deg,#103d32,#1b7053);padding:2rem 2.2rem;border-radius:18px;color:white;margin-bottom:1.2rem;}
.hero h1 {color:white;font-size:2.1rem;margin:0 0 .4rem 0;}
.hero p {color:#e5f2ec;margin:0;max-width:780px;}
.small-note {color:#66756f;font-size:.88rem;}
div[data-testid="stChatMessage"] {border:1px solid #e6ece8;border-radius:14px;padding:.65rem 1rem;}
.stButton button {border-radius:10px;}
</style>
<div class="hero"><h1>⚖️ Gambia Legal Chatbook</h1><p>Understand the laws, Acts, regulations and public policies of The Gambia — in clear, practical language, with sources you can check.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Chatbook settings")
    model = st.text_input("Ollama chat model", value=os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL))
    embedding = st.text_input("Embedding model", value=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    ollama_url = st.text_input("Ollama server", value=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    st.divider()
    store = LawStore()
    st.metric("Indexed passages", len(store.chunks))
    st.caption(store.status)
    st.markdown("**Primary research sources**")
    st.markdown("- [National Assembly — Acts & Laws](https://assembly.gm/documents/type/acts)")
    st.markdown("- [National Assembly — legislative documents](https://assembly.gm/documents)")
    st.markdown("- [Ministry of Justice — downloads and Gazette](https://www.moj.gm/downloads)")
    st.markdown("- [Judiciary — Constitution](https://judiciary.gov.gm/sites/default/files/2021-07/Constitution%20of%20The%20Gambia.pdf)")
    st.markdown("- [Law Hub Gambia — case law and legal materials](https://www.lawhubgambia.com/)")
    st.divider()
    st.markdown("**Add a legal document**")
    uploaded_pdf = st.file_uploader("Upload a PDF (for example, an Act or policy)", type=["pdf"], help="Only upload documents you are authorised to use. Uploaded files are labelled as user-supplied and are not automatically verified as official or current.")
    if uploaded_pdf and st.button("Index uploaded PDF", use_container_width=True):
        try:
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = "user_" + uploaded_pdf.name.replace("/", "_").replace("\\", "_")
            target = PDF_DIR / safe_name
            target.write_bytes(uploaded_pdf.getvalue())
            extracted = pdf_to_text(target).strip()
            if not extracted:
                st.error("No selectable text was found. This may be a scanned PDF; OCR is not yet included.")
            else:
                upload_store = LawStore()
                for idx, part in enumerate(chunk_text(extracted)):
                    if not upload_store.has_chunk(f"uploaded:{uploaded_pdf.name}", idx):
                        upload_store.add(Chunk(source=uploaded_pdf.name, source_url=f"uploaded:{uploaded_pdf.name}", chunk_index=idx, text=part, document_type="User-uploaded document — authenticity and legal status not verified"))
                try:
                    embed_client = OllamaClient(Config(chat_model=model, embedding_model=embedding, ollama_url=ollama_url))
                    for chunk in upload_store.chunks:
                        if chunk.source_url == f"uploaded:{uploaded_pdf.name}" and chunk.embedding is None:
                            chunk.embedding = embed_client.embed(chunk.text)
                except OllamaUnavailableError:
                    st.warning("Embeddings unavailable; the uploaded text is indexed for keyword retrieval.")
                upload_store.save()
                st.success(f"Indexed {len([c for c in upload_store.chunks if c.source_url == f'uploaded:{uploaded_pdf.name}'])} passage(s). Refresh the page to update the sidebar count.")
        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")
    if st.button("Clear this conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("**What can I help you understand?**")
examples = [
    "What does the Constitution say about freedom of expression?",
    "Explain the Labour Act and employee rights in simple terms.",
    "What is the difference between a Bill and an Act in The Gambia?",
    "What should I check before signing a tenancy agreement?",
]
cols = st.columns(2)
for i, prompt in enumerate(examples):
    if cols[i % 2].button(prompt, key=f"example_{i}", use_container_width=True):
        st.session_state.pending_question = prompt

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

question = st.chat_input("Ask a question about a Gambian law, Act, policy or legal process…")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1])
    config = Config(chat_model=model, embedding_model=embedding, ollama_url=ollama_url)
    with st.chat_message("assistant"):
        with st.spinner("Checking the indexed legal materials…"):
            try:
                answer = ask(config, question, history, LawStore())
            except OllamaUnavailableError as exc:
                answer = f"**The AI service could not be reached.** {exc}\n\nStart Ollama and ensure the selected models are installed."
            except Exception as exc:
                answer = f"Something went wrong while preparing the answer: `{exc}`"
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})

st.divider()
st.markdown("<p class='small-note'>Important: This tool provides general legal information, not legal advice. Its reliability depends on the documents indexed and their currency. Bills, draft policies and old copies must not be treated as current law without verification. For a decision affecting your rights, consult a qualified legal practitioner in The Gambia.</p>", unsafe_allow_html=True)
