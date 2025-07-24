#!/usr/bin/env python3
"""
groq_roleplay_agent.py
──────────────────────
Interactive **role-play** assistant powered by Groq’s Chat-Completions API.

Key features
------------
• System-prompt is externalised in JSON (default: ./groq_roleplay_config.json,  
  path overridable via GROQ_CONFIG_FILE).  
• Maintains full multi-turn chat history so the model stays in character.  
• Streams the assistant’s reply token-by-token for snappy UX.  
• Zero command-line flags: just run, type, and play.

Prerequisites
-------------
pip install requests
export GROQ_API_KEY="groq-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python groq_roleplay_agent.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final

import requests

# ─────────────────────────────── CONSTANTS ────────────────────────────────
API_URL:         Final[str] = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME:      Final[str] = "mixtral-8x7b-32768"        # solid general-purpose model
TIMEOUT_SECONDS: Final[int] = 90
CONFIG_FILE:     Final[Path] = Path(
    os.getenv("GROQ_CONFIG_FILE", "groq_roleplay_config.json")
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("❌  GROQ_API_KEY is not set—export it and re-run.")

HEADERS: Final[dict[str, str]] = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type":  "application/json",
}

# ────────────────────────── CONFIG → SYSTEM PROMPT ─────────────────────────
def _load_system_prompt() -> str:
    """
    Return the system prompt from CONFIG_FILE or fall back to a default.
    The prompt should instruct the model which persona to embody.
    """
    try:
        with CONFIG_FILE.open(encoding="utf-8") as fh:
            return json.load(fh)["system_prompt"]
    except Exception:
        # Sensible fallback: a flexible, game-style role-play guide
        return (
            "You are an improvisational actor.  Stay *in character* at all times, "
            "respond vividly, and drive the narrative forward.  Ask clarifying "
            "questions when needed, respect the user’s cues, and never break the "
            "fourth wall unless explicitly instructed."
        )


SYSTEM_PROMPT: Final[str] = _load_system_prompt()

# ────────────────────────── STREAMING CHAT CORE ───────────────────────────
def stream_chat(conversation: list[dict[str, str]]) -> str:
    """
    Send the full *conversation* to Groq, stream the assistant’s next message,
    return that complete assistant string for history tracking.
    """
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.8,            # a bit higher for creative role-play
        "max_completion_tokens": 4096,
        "top_p": 0.95,
        "stream": True,
        "messages": conversation,
    }

    assistant_chunks: list[str] = []

    with requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        stream=True,
        timeout=TIMEOUT_SECONDS,
    ) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue                               # keep-alive
            if line == b"data: [DONE]":
                print("\n⚡️  Stream complete\n")
                break

            chunk = json.loads(line.lstrip(b"data: ").decode("utf-8"))
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                assistant_chunks.append(token)
                print(token, end="", flush=True)

    return "".join(assistant_chunks)


# ──────────────────────────────── REPL ────────────────────────────────────
def main() -> None:
    # Initialise conversation with system prompt
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("🎭  Role-play agent ready (Ctrl-C to quit).")
    try:
        while True:
            user_input = input("> ").rstrip()
            if not user_input:
                continue
            conversation.append({"role": "user", "content": user_input})

            assistant_reply = stream_chat(conversation)
            conversation.append({"role": "assistant", "content": assistant_reply})
    except KeyboardInterrupt:
        print("\n👋  Bye!")


if __name__ == "__main__":
    main()
