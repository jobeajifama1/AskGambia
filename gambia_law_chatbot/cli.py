"""Command-line interface for the Gambia Law Chatbot."""

from __future__ import annotations

import argparse
import logging
import sys

from .chat import ask, chat
from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_URL,
    Config,
)
from .ingest import ingest
from .ollama_client import verify_ollama
from .store import LawStore

LOGGER = logging.getLogger("gambia_law_chatbot.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gambia_law_chatbot",
        description="Chat about Gambian laws in simple terms, locally via Ollama.",
    )
    parser.add_argument("--ingest", action="store_true", help="Fetch and index laws from government websites, then exit")
    parser.add_argument("--add", action="append", default=[], metavar="URL", help="Use this start URL instead of the defaults (repeatable)")
    parser.add_argument("question", nargs="*", help="One-off question instead of interactive chat")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"Ollama chat model (default: {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help=f"Ollama embedding model (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    config = Config.from_args(args)

    try:
        verify_ollama(config)
    except Exception as exc:
        LOGGER.warning("%s", exc)

    try:
        if args.ingest:
            ingest(config)
        elif args.question:
            print(ask(config, " ".join(args.question), "", LawStore()))
        else:
            chat(config)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
