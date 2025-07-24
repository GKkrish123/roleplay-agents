#!/usr/bin/env python3
"""
rag_annoy.py
────────────
Retrieval-Augmented Generation helper with:

  • 50 %-overlapping text chunks
  • OpenAI (o3) embeddings
  • Annoy for approximate nearest-neighbour search
  • Default on-disk persistence under ./data/

Install:
    pip install openai annoy tiktoken python-dotenv
Set env:
    export OPENAI_API_KEY="sk-…"

CLI usage:
    python rag_annoy.py docs/my_corpus.txt "What is ORI routing?"
"""

from __future__ import annotations

import os
import math
import json
import pathlib
from dataclasses import dataclass
from typing import List, Iterable

import openai                  # type: ignore
from annoy import AnnoyIndex   # type: ignore
import tiktoken                # type: ignore

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EMBED_MODEL       = "text-embedding-3-small"
INDEX_METRIC      = "angular"
CHUNK_TOKENS      = 512          # window size
OVERLAP_RATIO     = 0.5          # 50 %
TOP_K             = 4
ANN_SUFFIX        = ".ann"       # binary index file suffix
DEFAULT_DATA_DIR  = pathlib.Path(__file__).with_name("data")

# ──────────────────────────────────────────────
# Tokenisation helpers
# ──────────────────────────────────────────────
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _tokenise(text: str) -> List[int]:
    return _tokenizer.encode(text, disallowed_special=())


def _detokenise(tokens: List[int]) -> str:
    return _tokenizer.decode(tokens)


# ──────────────────────────────────────────────
# 50 %-overlap chunker
# ──────────────────────────────────────────────
def chunk_text(text: str,
               chunk_tokens: int = CHUNK_TOKENS,
               overlap_ratio: float = OVERLAP_RATIO) -> List[str]:
    if not 0 < overlap_ratio < 1:
        raise ValueError("overlap_ratio must be between 0 and 1 (exclusive)")

    tokens = _tokenise(text)
    step   = max(1, int(chunk_tokens * (1 - overlap_ratio)))
    chunks: List[str] = []

    for start in range(0, len(tokens), step):
        slice_tokens = tokens[start:start + chunk_tokens]
        if not slice_tokens:
            break
        chunks.append(_detokenise(slice_tokens))
        if start + chunk_tokens >= len(tokens):
            break
    return chunks


# ──────────────────────────────────────────────
# Embedding helper
# ──────────────────────────────────────────────
def embed_batch(lines: Iterable[str]) -> List[List[float]]:
    resp = openai.Embedding.create(input=list(lines), model=EMBED_MODEL)
    return [d["embedding"] for d in resp["data"]]


# ──────────────────────────────────────────────
# RAG class with Annoy index
# ──────────────────────────────────────────────
@dataclass(slots=True)
class RAGAnnoy:
    index: AnnoyIndex
    metadata: List[dict]  # index → {doc, chunk, text}

    # ------------------------ build ------------------------
    @classmethod
    def build(cls,
              docs: List[str],
              metric: str = INDEX_METRIC) -> "RAGAnnoy":
        all_chunks, metas = [], []
        for doc_id, text in enumerate(docs):
            for ch_id, chunk in enumerate(chunk_text(text)):
                all_chunks.append(chunk)
                metas.append({"doc": doc_id,
                              "chunk": ch_id,
                              "text": chunk})

        vectors = embed_batch(all_chunks)
        dim     = len(vectors[0])
        idx     = AnnoyIndex(dim, metric)
        for i, vec in enumerate(vectors):
            idx.add_item(i, vec)
        idx.build(int(math.sqrt(len(vectors))) or 2)

        return cls(idx, metas)

    # ------------------------ persistence ------------------
    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata, ensure_ascii=False))
        self.index.save(str(path.with_suffix(ANN_SUFFIX)))

    @classmethod
    def load(cls,
             path: str | pathlib.Path,
             metric: str = INDEX_METRIC) -> "RAGAnnoy":
        path      = pathlib.Path(path)
        metadata  = json.loads(path.read_text())
        # quick round-trip to get vector dim
        dim = len(openai.Embedding.create(input=["ping"], model=EMBED_MODEL)
                  ["data"][0]["embedding"])
        idx = AnnoyIndex(dim, metric)
        idx.load(str(path.with_suffix(ANN_SUFFIX)))
        return cls(idx, metadata)

    # ------------------------ retrieval --------------------
    def retrieve(self, query: str, k: int = TOP_K) -> List[dict]:
        q_vec = embed_batch([query])[0]
        ids   = self.index.get_nns_by_vector(q_vec, k, include_distances=False)
        return [self.metadata[i] for i in ids]

    # ------------------------ generation ------------------
    def generate_answer(self, query: str,
                        system_prompt: str | None = None,
                        k: int = TOP_K) -> str:
        ctx_chunks = self.retrieve(query, k)
        context    = "\n\n".join(ch["text"] for ch in ctx_chunks)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user",
                         "content": f"Context:\n{context}\n\nQuestion: {query}"})

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=messages,
        )
        return resp["choices"][0]["message"]["content"].strip()


# ──────────────────────────────────────────────
# CLI smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="RAG demo with 50 % overlap + Annoy index."
    )
    parser.add_argument("corpus",
                        help="Plain-text file to ingest")
    parser.add_argument("query",
                        help="Question to ask")
    parser.add_argument("--save", default=None,
                        help="Save index metadata as JSON "
                             "(default: ./data/rag_index.json)")
    args = parser.parse_args()

    # OpenAI credentials
    openai.api_key  = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE") or openai.api_base

    corpus_text = pathlib.Path(args.corpus).read_text(encoding="utf-8")
    rag         = RAGAnnoy.build([corpus_text])

    # -------- persistence ----------
    save_path = pathlib.Path(args.save) if args.save else (
        DEFAULT_DATA_DIR / "rag_index.json"
    )
    rag.save(save_path)
    print(f"💾 Index persisted under {save_path}")

    # -------- answer ---------------
    answer = rag.generate_answer(args.query)
    print("\n" + textwrap.fill(answer, 110))
