"""Grounded question answering for Gambian legal and policy materials."""
from __future__ import annotations
from .config import FOLLOWUP_PROMPT, SYSTEM_PROMPT, Config, OllamaUnavailableError
from .ollama_client import OllamaClient
from .store import Chunk, LawStore

def build_context(records: list[Chunk]) -> str:
    return "\n\n".join(
        f"### {r.source}\nDocument type/status: {r.document_type}\nPublication date: {r.publication_date}\nURL: {r.source_url}\nPassage {r.chunk_index + 1}:\n{r.text}"
        for r in records
    )

def refine_query(client: OllamaClient, question: str, history: str) -> str:
    if not history: return question
    try:
        refined = client.generate(FOLLOWUP_PROMPT + f"\n\nConversation:\n{history[-5000:]}\n\nLatest question: {question}", temperature=0.0)
        return refined or question
    except OllamaUnavailableError: return question

def ask(config: Config, question: str, history: str, store: LawStore) -> str:
    client = OllamaClient(config)
    query = refine_query(client, question, history)
    records = store.retrieve(query, config)
    if records:
        prompt = f"{SYSTEM_PROMPT}\n\nUSER QUESTION:\n{question}\n\nRETRIEVED OFFICIAL / REFERENCE MATERIALS:\n---\n{build_context(records)}\n---\nAnswer using the required format. Do not cite anything outside these excerpts."
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nUSER QUESTION:\n{question}\n\nNo sufficiently relevant passage was found in the local document library. Do not guess the law. Say that the current legal position could not be verified from the indexed materials, suggest the most relevant official source or authority to check, and ask for the missing detail if a narrower search would help."
    return client.generate(prompt, temperature=0.1)

def chat(config: Config) -> None:
    store = LawStore(); print("\nGAMBIA LEGAL CHATBOOK | Ask about Gambian laws, Acts and policies\nType 'quit' to exit.\n")
    print(store.status, "\n"); history = ""
    while True:
        try: question = input("You: ").strip()
        except (EOFError,KeyboardInterrupt): print("\nGoodbye!"); break
        if question.lower() in {"quit","exit","q"}: print("Goodbye!"); break
        if not question: continue
        try: answer = ask(config, question, history, store)
        except OllamaUnavailableError as exc: answer = f"The local AI service is unavailable: {exc}"
        print(f"\nChatbook: {answer}\n")
        history += f"\nUser: {question}\nAssistant: {answer}\n"
