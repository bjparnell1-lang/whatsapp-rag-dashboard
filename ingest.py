"""WhatsApp transcript RAG ingestion (working copy in /tmp)."""
from __future__ import annotations
import argparse, os, re, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding import build_embedder, TfidfSvdEmbeddingFunction

DEFAULT_ROOTS = ["/tmp/wa_extracted",
                 "/sessions/beautiful-awesome-albattani/mnt/Whats App"]
def _default_db_path():
    return "/tmp/whatsapp_vector_db" if Path("/sessions").exists() \
        else str(Path(__file__).resolve().parent / "whatsapp_vector_db")
DB_PATH = _default_db_path()
COLLECTION_NAME = "whatsapp_messages"

LINE_RE = re.compile(
    r"^‎?\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]\s*"
    r"(?P<sender>[^:]+?):\s?(?P<msg>.*)$"
)
VOICE = [re.compile(r"\.opus", re.I),
         re.compile(r"audio omitted", re.I),
         re.compile(r"voice message omitted", re.I),
         re.compile(r"voice note", re.I),
         re.compile(r"<attached:[^>]*AUDIO[^>]*>", re.I)]
CALL  = [re.compile(r"voice call", re.I),
         re.compile(r"video call", re.I),
         re.compile(r"missed voice call", re.I),
         re.compile(r"missed video call", re.I)]
MEDIA = [re.compile(r"image omitted", re.I),
         re.compile(r"video omitted", re.I),
         re.compile(r"sticker omitted", re.I),
         re.compile(r"document omitted", re.I),
         re.compile(r"GIF omitted", re.I),
         re.compile(r"Contact card omitted", re.I)]

@dataclass
class Message:
    line_no: int; raw: str; date: str; sender: str; text: str
    is_voice: bool=False; is_call: bool=False; is_media: bool=False

def _clean(s): return re.sub(r"^[‎‏‪-‮﻿]+","",s).rstrip("\n")
def _classify(t):
    return (any(p.search(t) for p in VOICE),
            any(p.search(t) for p in CALL),
            any(p.search(t) for p in MEDIA))
def _parse_date(d,t):
    raw=f"{d} {t}".strip()
    for fmt in ["%m/%d/%y %I:%M:%S %p","%m/%d/%Y %I:%M:%S %p",
                "%m/%d/%y %I:%M %p","%m/%d/%Y %I:%M %p",
                "%m/%d/%y %H:%M:%S","%m/%d/%Y %H:%M:%S"]:
        try: return datetime.strptime(raw,fmt).isoformat()
        except ValueError: continue
    return raw

def parse_file(path):
    msgs=[]; cur=None
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for idx,raw in enumerate(f,1):
            line=_clean(raw)
            if not line: continue
            m=LINE_RE.match(line)
            if m:
                if cur: msgs.append(cur)
                v,c,md=_classify(_clean(m.group("msg")))
                cur=Message(idx,line,_parse_date(m.group("date"),m.group("time")),
                            m.group("sender").strip(),_clean(m.group("msg")),v,c,md)
            else:
                if cur is not None:
                    cur.text=(cur.text+"\n"+line).strip()
                    v,c,md=_classify(cur.text)
                    cur.is_voice|=v; cur.is_call|=c; cur.is_media|=md
    if cur: msgs.append(cur)
    return msgs

def window_chunks(msgs, window=8, stride=4):
    if not msgs: return
    n=len(msgs)
    for start in range(0,n,stride):
        end=min(start+window,n)
        yield msgs[start:end], start
        if end>=n: break

def render_chunk(w):
    parts=[]
    for m in w:
        mk=" [VOICE_NOTE]" if m.is_voice else (" [CALL_EVENT]" if m.is_call else (" [MEDIA]" if m.is_media else ""))
        parts.append(f"[{m.date}] {m.sender}{mk}: {m.text}")
    return "\n".join(parts)

def chunk_metadata(w, src, root):
    folder=src.parent.name
    senders=sorted({m.sender for m in w})
    return {"folder_source":folder,
            "chat_name":folder.replace("__"," | "),
            "file":str(src),
            "senders":" | ".join(senders),
            "start_date":w[0].date,"end_date":w[-1].date,
            "start_line":w[0].line_no,"end_line":w[-1].line_no,
            "has_voice":"1" if any(m.is_voice for m in w) else "0",
            "has_call":"1" if any(m.is_call for m in w) else "0",
            "has_media":"1" if any(m.is_media for m in w) else "0"}

def find_root():
    for r in DEFAULT_ROOTS:
        if Path(r).is_dir(): return Path(r)
    raise SystemExit("no root")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=None)
    ap.add_argument("--db",default=DB_PATH)
    ap.add_argument("--collection",default=COLLECTION_NAME)
    ap.add_argument("--window",type=int,default=8)
    ap.add_argument("--stride",type=int,default=4)
    ap.add_argument("--reset",action="store_true")
    ap.add_argument("--embedder",choices=["tfidf","openai"],default="tfidf")
    args=ap.parse_args()

    root=Path(args.root) if args.root else find_root()
    print(f"[ingest] root = {root}",flush=True)
    txts=[Path(d)/f for d,_,fs in os.walk(root) for f in fs if f.lower().endswith(".txt")]
    print(f"[ingest] found {len(txts)} .txt files",flush=True)
    if not txts: sys.exit("no .txt")

    os.makedirs(args.db,exist_ok=True)
    docs=[]; metas=[]; ids=[]; total=0
    for p in txts:
        ms=parse_file(p); total+=len(ms)
        print(f"[ingest] {p.parent.name}/{p.name} -> {len(ms)} msgs",flush=True)
        for w,_ in window_chunks(ms,args.window,args.stride):
            if not w: continue
            docs.append(render_chunk(w))
            metas.append(chunk_metadata(w,p,root))
            ids.append(f"{p.parent.name}::{p.name}::{w[0].line_no}-{w[-1].line_no}")
    print(f"[ingest] parsed {total} msgs -> {len(docs)} chunks",flush=True)
    if not docs: sys.exit("zero chunks")

    embedder=build_embedder(args.db,kind=args.embedder)
    if isinstance(embedder, TfidfSvdEmbeddingFunction) and embedder.vectorizer is None:
        print("[ingest] fitting TF-IDF + SVD...",flush=True)
        embedder.fit(docs)
        print(f"[ingest] embedder fit -> {embedder.path}",flush=True)

    print("[ingest] opening chromadb client...",flush=True)
    client=chromadb.PersistentClient(path=args.db,settings=Settings(anonymized_telemetry=False))
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"[ingest] dropped {args.collection}",flush=True)
        except Exception: pass
    col=client.get_or_create_collection(name=args.collection,
                                        embedding_function=embedder,
                                        metadata={"hnsw:space":"cosine"})
    print("[ingest] adding chunks...",flush=True)
    BATCH=256
    for i in range(0,len(docs),BATCH):
        col.add(documents=docs[i:i+BATCH],metadatas=metas[i:i+BATCH],ids=ids[i:i+BATCH])
        print(f"[ingest] added {min(i+BATCH,len(docs))}/{len(docs)}",flush=True)
    print(f"[ingest] DONE count={col.count()} db={args.db}",flush=True)

if __name__=="__main__":
    main()
