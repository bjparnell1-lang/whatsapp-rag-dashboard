"""
WhatsApp transcript RAG retrieval + GPT-4o audit.

Usage:
    python ask.py "every time Santigie refused to pay"
    python ask.py "voice notes around late April 2026" --n 50
    python ask.py "..." --no-llm        # just dump raw chunks
    python ask.py "..." --chat "Joseph bK Caller"   # filter by folder substring

The retriever pulls up to --n chunks (default 40) from the local ChromaDB,
sorts them chronologically, and hands them to GPT-4o with an "objective
auditor" system prompt. Every fact in the answer must cite [Sender, Date].
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding import TfidfSvdEmbeddingFunction, build_embedder

def default_db():
    return "/tmp/whatsapp_vector_db" if Path("/sessions").exists() \
        else str(Path(__file__).resolve().parent / "whatsapp_vector_db")

SYSTEM_PROMPT = """You are an objective forensic auditor reading WhatsApp transcript excerpts.
Your job: compile EVERY single instance in the provided excerpts that matches the user's request.
Rules:
- Order findings strictly chronologically (earliest -> latest).
- Each finding MUST end with an inline citation in this exact format:  [Sender, YYYY-MM-DD HH:MM]
- Quote the relevant phrase verbatim (in double quotes) when it carries the evidence.
- If a finding is a voice note, audio attachment, or call, mark it with the tag (VOICE NOTE) or (CALL) before the citation.
- Do NOT speculate; if the excerpts do not contain something, say so.
- If multiple chats contain matches, group them by chat under a "## <chat name>" header, still chronological within each group.
- End with a short "## Summary" of patterns, counts, and gaps.
"""

def query(args):
    embedder = TfidfSvdEmbeddingFunction(args.db)
    if embedder.vectorizer is None:
        sys.exit(f"No fitted embedder at {args.db}. Run ingest.py first.")
    client = chromadb.PersistentClient(path=args.db,
                                       settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(args.collection, embedding_function=embedder)

    where = None
    if args.chat:
        # Chroma where-clause; substring isn't supported, so we filter post-hoc too.
        where = None
    if args.has_voice:
        where = {"has_voice": "1"}
    if args.has_call:
        where = {"has_call": "1"}

    res = col.query(query_texts=[args.query],
                    n_results=args.n,
                    where=where,
                    include=["documents","metadatas","distances"])

    hits = []
    for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if args.chat and args.chat.lower() not in (m.get("chat_name","") + m.get("folder_source","")).lower():
            continue
        hits.append({"doc": d, "meta": m, "distance": dist})

    # Sort chronologically by start_date
    hits.sort(key=lambda h: h["meta"].get("start_date",""))
    return hits

def dump_context(hits) -> str:
    blocks = []
    for h in hits:
        m = h["meta"]
        header = f"### chat={m['chat_name']} | lines {m['start_line']}-{m['end_line']} | dist={h['distance']:.3f}"
        blocks.append(header + "\n" + h["doc"])
    return "\n\n".join(blocks)

def call_llm(question, hits, model="gpt-4o"):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY env var is required for the GPT-4o pass.\n"
                 "Re-run with --no-llm to dump raw retrieved chunks instead.")
    client = OpenAI(api_key=api_key)
    user_msg = (f"USER QUESTION:\n{question}\n\n"
                f"EXCERPTS (already pre-sorted chronologically, {len(hits)} chunks):\n\n"
                + dump_context(hits))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user_msg}],
        temperature=0.0,
    )
    return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Natural-language question")
    ap.add_argument("--db", default=default_db())
    ap.add_argument("--collection", default="whatsapp_messages")
    ap.add_argument("-n","--n", type=int, default=40, help="chunks to retrieve")
    ap.add_argument("--chat", default=None, help="filter chunks by chat-name substring")
    ap.add_argument("--has-voice", action="store_true", help="only chunks containing voice notes")
    ap.add_argument("--has-call",  action="store_true", help="only chunks containing call events")
    ap.add_argument("--no-llm", action="store_true", help="skip GPT-4o; print raw chunks")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--json", action="store_true", help="dump hits as JSON instead of prose")
    args = ap.parse_args()

    hits = query(args)
    print(f"[ask] retrieved {len(hits)} chunks", file=sys.stderr)

    if args.json:
        print(json.dumps(hits, indent=2, default=str))
        return
    if args.no_llm:
        print(dump_context(hits))
        return
    print(call_llm(args.query, hits, model=args.model))

if __name__ == "__main__":
    main()
