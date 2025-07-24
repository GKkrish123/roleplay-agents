#!/usr/bin/env python3
"""
auto_research_agent.py
──────────────────────
Autonomous research agent that:

1. Uses **SerpAPI** to gather the top N search results
2. Downloads each page, extracts human-readable text (BeautifulSoup)
3. Summarises every source with Groq’s LLM (streaming)
4. Produces a final synthesis that cites the individual summaries

────────────────────────────  QUICK START  ────────────────────────────
pip install requests beautifulsoup4
export SERPAPI_KEY="serp-xxxxxxxxxxxxxxxxxxxxxxxx"
export GROQ_API_KEY="groq-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python auto_research_agent.py
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Final, List, Tuple

import requests
from bs4 import BeautifulSoup

# ───────────────────────────── CONFIG ─────────────────────────────────
SERP_ENDPOINT: Final[str] = "https://serpapi.com/search.json"
GROQ_ENDPOINT: Final[str] = "https://api.groq.com/openai/v1/chat/completions"
# MODEL_NAME:    Final[str] = "mixtral-8x7b-32768"
MODEL_NAME:    Final[str] = "llama-3.3-70b-versatile"
TIMEOUT:       Final[int] = 45
MAX_ARTICLE_CHARS: Final[int] = 10_000          # truncate huge pages

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not (SERPAPI_KEY and GROQ_API_KEY):
    sys.exit("❌  Set SERPAPI_KEY and GROQ_API_KEY env vars first.")

# Optional external prompt config
CONFIG_FILE = Path(os.getenv("RESEARCH_CONFIG_FILE", "research_config.json"))
def _load_prompt(key: str, fallback: str) -> str:
    try:
        with CONFIG_FILE.open(encoding="utf-8") as f:
            return json.load(f)[key]
    except Exception:
        return fallback

SYS_PROMPT_SINGLE = _load_prompt(
    "single_source_prompt",
    (
        "You are a world-class researcher. Summarise the following article in "
        "4–6 bullet points focused on facts relevant to the user’s question, "
        "then end with 'SOURCE:' followed by the page title."
    ),
)
SYS_PROMPT_FINAL = _load_prompt(
    "synthesis_prompt",
    (
        "Using the bullet-point findings listed below, synthesize a concise, "
        "coherent answer for the user. Provide a short paragraph, then a list "
        "of takeaway bullets. Finish with 'CITATIONS:' and the numbered list "
        "of sources referenced."
    ),
)

HEADERS_GROQ = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# ─────────────────────────── SERPAPI SEARCH ──────────────────────────
def serp_search(query: str, num_results: int = 5) -> List[Tuple[str, str]]:
    """Return (title, url) tuples for the top *num_results* links."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
    }
    r = requests.get(SERP_ENDPOINT, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    links = []
    for res in data.get("organic_results", [])[:num_results]:
        url = res.get("link")
        title = html.unescape(res.get("title", ""))
        if url:
            links.append((title, url))
    return links

# ───────────────────────────  UTILITIES  ─────────────────────────────
_RE_NEWLINES = re.compile(r"\n{2,}")
def clean_html(html_text: str) -> str:
    """Strip script/style tags and collapse whitespace."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = _RE_NEWLINES.sub("\n", text)
    return text.strip()

# ───────────────────────── PAGE DOWNLOAD & TRIM ──────────────────────
def fetch_article(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=TIMEOUT)
    r.raise_for_status()
    return clean_html(r.text)[:MAX_ARTICLE_CHARS]

# ────────────────────────── LLM SUMMARISATION ────────────────────────
def llm_chat(system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "temperature": 0.6,
        "max_completion_tokens": 4096,
        "top_p": 0.95,
        "stream": False,
        "stop": None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    r = requests.post(GROQ_ENDPOINT, headers=HEADERS_GROQ, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

from typing import Tuple

# ────────────────────────── CORE RESEARCH FLOW ───────────────────────
def auto_research(question: str, k: int = 5) -> Tuple[str, str]:
    print(f"🔍  Searching for: {question!r}")
    sources = serp_search(question, num_results=k)
    if not sources:
        print("No results found.")
        return "No results found.", "No results found."

    bullet_points = []   # str list of summaries
    source_refs  = []    # numbered (title, url)
    for idx, (title, url) in enumerate(sources, 1):
        print(f"\n[{idx}/{k}]  Fetching {title} …")
        try:
            text = fetch_article(url)
        except Exception as e:
            print(f"   ✖  Skip ({e})")
            continue

        summary = llm_chat(
            SYS_PROMPT_SINGLE,
            f"User question: {question}\n\nArticle text:\n{text}",
        )
        bullet_points.append(f"{idx}. {summary}")
        source_refs.append(f"[{idx}] {title} – {url}")
        print("   ✔  Summarised.")

        # Gentle throttle to respect target sites
        time.sleep(1.2)

    if not bullet_points:
        print("Could not summarise any sources.")
        return "Could not summarise any sources.", "Could not summarise any sources."

    combined = "\n\n".join(bullet_points)
    final_answer = llm_chat(
        SYS_PROMPT_FINAL,
        combined,
        temperature=0.3,
    )

    print("\n" + "=" * 80)
    print("📚  FINAL SYNTHESIS\n")
    print(final_answer)
    print("\n" + "\n".join(source_refs))
    return final_answer, combined

# ─────────────────────────────  CLI LOOP  ─────────────────────────────
def main() -> None:
    print("🧠  Autonomous Research Agent (Ctrl-C to quit)")
    try:
        while True:
            q = input("\n❓  Enter your question: ").strip()
            if q:
                auto_research(q)
    except KeyboardInterrupt:
        print("\n👋  Bye!")

if __name__ == "__main__":
    main()
