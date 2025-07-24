#!/usr/bin/env python3
"""
master_agent.py
──────────────────
Unified Groq-powered assistant + DataFrame sandbox.

Dependencies
------------
pip install requests pandas numpy beautifulsoup4

Environment
-----------
export GROQ_API_KEY="groq-…"
export SERPAPI_KEY="serp-…"      # needed only for research

Usage
-----
python thinking_agent.py
> load ~/data/sales.csv
> df.head()
> math: integrate x^2 from 0 to 2
> code: write a BFS in Go
> research: global EV market share 2024
Ctrl-C to quit.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# ─────────────────────────── External Agents ───────────────────────────
# Math solver (new import path per user request)
try:
    from groq_math import solve_math                # ← only change vs previous design
except ImportError as e:
    raise ImportError("groq_math.solve_math not found on PYTHONPATH") from e

# Coding assistant
try:
    from groq_coding_agent import stream_chat as coding_chat
except ImportError as e:
    raise ImportError("groq_coding_agent.stream_chat not found") from e

# Research agent
try:
    from auto_research_agent import auto_research
except ImportError as e:
    raise ImportError("auto_research_agent.auto_research not found") from e

# Role-play agent is optional – hook left for future extension
# from groq_roleplay_agent import stream_chat as roleplay_chat

# ───────────────────────────── Heuristics ──────────────────────────────
_REGEX_MATH     = re.compile(r"\b(integrate|∫|derivative|limit|solve|root|dx)\b", re.I)
_REGEX_CODE     = re.compile(r"\b(class|def|algorithm|complexity|python|golang|java|csharp)\b", re.I)

df: pd.DataFrame | None = None          # live DataFrame (global)

# ─────────────────────── DataFrame utilities ───────────────────────────
def load_dataframe(path: str) -> None:
    """Load CSV/TSV/Parquet into global `df`."""
    global df
    p = Path(path).expanduser()
    if not p.exists():
        print(f"❌  File not found: {p}")
        return

    try:
        if p.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(p, sep=sep)
        elif p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            print("❌  Unsupported file type.")
            return
        print(f"✅  DataFrame loaded: {len(df):,} rows × {len(df.columns)} cols")
    except Exception as exc:               # noqa: BLE001
        print(f"❌  Failed to load file: {exc}")

def eval_dataframe(expr: str) -> None:
    """Safely evaluate a Pandas expression like 'df.describe()'."""
    if df is None:
        print("❌  No DataFrame loaded. Use `load <file>` first.")
        return
    try:
        # Restricted namespace: df / pd / np only
        result = eval(expr, {"__builtins__": {}}, {"df": df, "pd": pd, "np": np})
        if isinstance(result, pd.DataFrame) and result.size <= 200:
            print(result.to_string(index=False))
        else:
            print(result)
    except Exception as exc:               # noqa: BLE001
        print(f"❌  DataFrame error: {exc}")

# ─────────────────────────── Dispatchers ───────────────────────────────
def handle_math(query: str) -> None:
    try:
        print(solve_math(query))
    except Exception as exc:               # noqa: BLE001
        print(f"❌  Math agent error: {exc}")

def handle_code(prompt: str) -> None:
    try:
        coding_chat(prompt)                # streams internally
    except Exception as exc:
        print(f"❌  Coding agent error: {exc}")

def handle_research(question: str) -> None:
    try:
        auto_research(question, k=5)       # prints internally
    except Exception as exc:
        print(f"❌  Research agent error: {exc}")

# ──────────────────────────── Router ───────────────────────────────────
def dispatch(user_input: str) -> None:
    s = user_input.strip()

    # DataFrame commands
    if s.startswith("load "):
        load_dataframe(s[5:].strip())
        return
    if s.startswith("df."):
        eval_dataframe(s)
        return

    # Explicit prefixes
    if s.startswith("math:"):
        handle_math(s[5:].strip())
        return
    if s.startswith("code:"):
        handle_code(s[5:].strip())
        return
    if s.startswith("research:"):
        handle_research(s[9:].strip())
        return

    # Heuristic routing
    if _REGEX_MATH.search(s):
        handle_math(s)
    elif _REGEX_CODE.search(s):
        handle_code(s)
    else:
        handle_research(s)

# ───────────────────────────── REPL ────────────────────────────────────
def main() -> None:
    print("🛠️  Thinking Agent ready.  Commands: load <file>, df.<expr>, math:, code:, research:")
    try:
        while True:
            user_input = input("> ").rstrip()
            if user_input:
                dispatch(user_input)
    except KeyboardInterrupt:
        print("\n👋  Bye!")

if __name__ == "__main__":
    main()
