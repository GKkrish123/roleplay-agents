#!/usr/bin/env python3
"""
groq_stream_client.py
─────────────────────
Production-grade Groq Chat-Completions streaming client.

• Reads your system-prompt from a JSON config file (default: ./groq_config.json).
• Streams the model’s response token-by-token to stdout.
• Runs in an interactive REPL loop—type a prompt, get an answer, repeat.

Prerequisites
-------------
pip install requests
export GROQ_API_KEY="groq-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final

import requests

# ───────────────────────── Configuration ──────────────────────────────────────
API_URL:          Final[str] = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME:       Final[str] = "deepseek-r1-distill-llama-70b"
TIMEOUT_SECONDS:  Final[int] = 60
CONFIG_FILE:      Final[Path] = Path(os.getenv("GROQ_CONFIG_FILE", "groq_think_agent_config.json"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("❌  GROQ_API_KEY is not set—export it and re-run.")

HEADERS: Final[dict[str, str]] = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type":  "application/json",
}

# ──────────────────────── Helper Functions ────────────────────────────────────
def _load_system_prompt() -> str:
    """
    Return the system prompt from CONFIG_FILE.
    Falls back to a safe default if file/key is missing or malformed.
    """
    try:
        with CONFIG_FILE.open(encoding="utf-8") as fh:
            return json.load(fh)["system_prompt"]
    except Exception:
        return (
            "You are a meticulous mathematician. "
            "Solve the user's problem in clear, numbered steps, "
            "then finish with:\nFINAL ANSWER: <answer>"
        )


SYSTEM_PROMPT: Final[str] = _load_system_prompt()

from typing import Tuple

def stream_chat(user_prompt: str, *, system_prompt: str = SYSTEM_PROMPT) -> Tuple[str, str]:
    """
    Send *user_prompt* to Groq and stream the reply to stdout.
    """
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.6,
        "max_completion_tokens": 4096,
        "top_p": 0.95,
        "stream": True,
        "stop": None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }

    with requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        stream=True,
        timeout=TIMEOUT_SECONDS,
    ) as resp:
        resp.raise_for_status()
        explanation = ""
        for line in resp.iter_lines():
            if not line:                      # Ignore keep-alive blanks
                continue
            if line == b"data: [DONE]":       # End of stream
                print("\n⚡️  Stream complete\n")
                break

            chunk = json.loads(line.lstrip(b"data: ").decode("utf-8"))
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
                explanation += delta["content"]
        return explanation.split("**FINAL ANSWER:** ")[1], explanation


# ──────────────────────────── REPL Loop ───────────────────────────────────────
def main() -> None:
    print("📝  Enter a prompt (Ctrl-C to quit):")
    try:
        while True:
            prompt = input("> ").strip()
            if prompt:
                stream_chat(prompt)
    except KeyboardInterrupt:
        print("\n👋  Bye!")


if __name__ == "__main__":
    main()
