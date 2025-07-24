#!/usr/bin/env python3
"""
groq_coding_agent.py
────────────────────
Production-ready *coding* assistant that streams Groq Chat-Completions.

• Reads a **coding-focused system prompt** from a JSON config file  
  (default: ./groq_coding_config.json, overridable via GROQ_CONFIG_FILE).  
• Opens an interactive REPL — type a coding question / request, receive
  the answer token-by-token in real time.  
• Depends only on `requests`.

────────────────────────────  QUICK START  ────────────────────────────
pip install requests
export GROQ_API_KEY="groq-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python groq_coding_agent.py
───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final, Tuple

import requests

# ─────────────────────────── CONSTANTS ───────────────────────────────
API_URL:         Final[str] = "https://api.groq.com/openai/v1/chat/completions"
# MODEL_NAME:      Final[str] = "codellama-70b-instruct"          # coding-oriented model
# MODEL_NAME:      Final[str] = "meta-llama/llama-4-scout-17b-16e-instruct"          # coding-oriented model
MODEL_NAME:      Final[str] = "qwen-qwq-32b"          # coding-oriented model
TIMEOUT_SECONDS: Final[int] = 90
CONFIG_FILE:     Final[Path] = Path(
    os.getenv("GROQ_CONFIG_FILE", "groq_coding_config.json")
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("❌  GROQ_API_KEY is not set—export it and re-run.")

HEADERS: Final[dict[str, str]] = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type":  "application/json",
}

# ───────────────────── CONFIG → SYSTEM PROMPT ───────────────────────
def _load_system_prompt() -> str:
    """
    Pull `"system_prompt"` out of CONFIG_FILE.
    Falls back to a sensible default if file / key is missing.
    """
    try:
        with CONFIG_FILE.open(encoding="utf-8") as fh:
            return json.load(fh)["system_prompt"]
    except Exception:
        return (
            "You are a senior software engineer.  Produce **production-quality** "
            "code with minimal dependencies, clear comments, and best practices. "
            "If you modify existing code, highlight diffs.  Answer only with code "
            "and concise explanations."
        )


SYSTEM_PROMPT: Final[str] = _load_system_prompt()

# ───────────────────────── STREAMING CORE ────────────────────────────
def stream_chat(prompt: str, *, system_prompt: str = SYSTEM_PROMPT) -> Tuple[str, str]:
    """
    Send *prompt* to Groq and stream the reply token-by-token to stdout.
    """
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.15,
        "max_completion_tokens": 4096,
        "top_p": 0.95,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
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
        content = ""
        for line in resp.iter_lines():
            if not line:
                continue                       # keep-alive
            if line == b"data: [DONE]":
                print("\n⚡️  Stream complete\n")
                break

            chunk = json.loads(line.lstrip(b"data: ").decode("utf-8"))
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                print(delta["content"], end="", flush=True)
                content += delta["content"]
        return content, content
# ────────────────────────────── REPL ────────────────────────────────
def main() -> None:
    print("💻  Coding agent ready (Ctrl-C to quit).")
    try:
        while True:
            user_input = input("> ").rstrip()
            if user_input:
                stream_chat(user_input)
    except KeyboardInterrupt:
        print("\n👋  Bye!")


if __name__ == "__main__":
    main()
