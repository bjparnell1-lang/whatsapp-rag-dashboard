"""
Streamlit UI wrapper for the WhatsApp forensic RAG.

Run locally:
    cd "C:\\Users\\bjpar\\Documents\\Whats App\\rag"
    pip install streamlit reportlab pandas
    set OPENAI_API_KEY=sk-...        (only needed for the AI Audit mode)
    streamlit run app.py

This file does NOT touch ingest.py or the embedder. It reuses ask.py's
`query()`, `dump_context()`, and `call_llm()` against the existing
ChromaDB at ./whatsapp_vector_db.
"""
from __future__ import annotations
import os
# MUST be set before chromadb / opentelemetry imports. Works around a protobuf
# descriptor mismatch when chromadb's bundled _pb2.py files run under newer
# protobuf (e.g. Python 3.14 on Streamlit Cloud).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import io
import sys
import csv
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Hoist Streamlit Cloud secrets into os.environ so ask.call_llm() can read them.
try:
    if "OPENAI_API_KEY" in st.secrets and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

import ask
import chromadb
from chromadb.config import Settings
from embedding import TfidfSvdEmbeddingFunction

# Folder inside the repo that mirrors each chat's source .txt for context view.
LOCAL_CHATS_ROOT = Path(__file__).resolve().parent / "chats"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = ask.default_db()
COLLECTION = "whatsapp_messages"


def _account_owner_from_file(file_path: str) -> str:
    """Derive an 'account owner' label from the chat file path.
    `<root>/<account_owner>/<folder_source>/_chat.txt`  -> account_owner
    Falls back to '(root)' if there is no extra layer.
    """
    try:
        p = Path(file_path)
        return p.parent.parent.name or "(root)"
    except Exception:
        return "(unknown)"


@st.cache_resource(show_spinner=False)
def open_collection():
    embedder = TfidfSvdEmbeddingFunction(DB_PATH)
    if embedder.vectorizer is None:
        raise FileNotFoundError(
            f"No fitted embedder at {DB_PATH}. Run `python ingest.py` first."
        )
    client = chromadb.PersistentClient(
        path=DB_PATH, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(COLLECTION, embedding_function=embedder)


@st.cache_data(show_spinner="Indexing chats and date range...")
def load_facets():
    """Walk metadata to compute facets for the sidebar."""
    col = open_collection()
    total = col.count()
    # Pull metadatas in pages so we don't blow up memory.
    chats = set()
    owners = set()
    earliest, latest = None, None
    BATCH = 1000
    fetched = 0
    while fetched < total:
        res = col.get(limit=BATCH, offset=fetched, include=["metadatas"])
        metas = res.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            chats.add(m.get("chat_name", ""))
            owners.add(_account_owner_from_file(m.get("file", "")))
            for k in ("start_date", "end_date"):
                v = m.get(k, "")
                try:
                    d = datetime.fromisoformat(v).date()
                except Exception:
                    continue
                if earliest is None or d < earliest:
                    earliest = d
                if latest is None or d > latest:
                    latest = d
        fetched += len(metas)
    return {
        "chats": sorted(c for c in chats if c),
        "owners": sorted(o for o in owners if o),
        "earliest": earliest or date(2020, 1, 1),
        "latest": latest or date.today(),
        "total": total,
    }


def run_query(question, n, chat_names, owner_names, has_voice, has_call, has_media,
              start_d, end_d):
    """Run ask.query() then apply post-hoc filters (chat/owner/date/has_media)."""
    args = SimpleNamespace(
        db=DB_PATH,
        collection=COLLECTION,
        query=question,
        n=n,
        chat=None,
        has_voice=bool(has_voice),
        has_call=bool(has_call),
    )
    hits = ask.query(args)

    chats_set = set(chat_names or [])
    owners_set = set(owner_names or [])

    def keep(h):
        m = h["meta"]
        if chats_set and m.get("chat_name", "") not in chats_set:
            return False
        if owners_set and _account_owner_from_file(m.get("file", "")) not in owners_set:
            return False
        if has_media and m.get("has_media") != "1":
            return False
        try:
            sd = datetime.fromisoformat(m.get("start_date", "")).date()
        except Exception:
            sd = None
        try:
            ed = datetime.fromisoformat(m.get("end_date", "")).date()
        except Exception:
            ed = None
        # Keep if chunk window intersects the requested date range.
        anchor = sd or ed
        if anchor is None:
            return True
        if start_d and (ed or anchor) < start_d:
            return False
        if end_d and (sd or anchor) > end_d:
            return False
        return True

    return [h for h in hits if keep(h)]


def _resolve_chat_file(file_path: str, folder_source: str) -> Path | None:
    """Find the .txt for context view.
    Tries the original (Windows) path first, then `chats/<folder_source>/_chat.txt`
    inside the repo so it works on Streamlit Cloud (Linux).
    """
    p = Path(file_path)
    if p.exists():
        return p
    if folder_source:
        candidate = LOCAL_CHATS_ROOT / folder_source / "_chat.txt"
        if candidate.exists():
            return candidate
    # Last resort: match by basename anywhere under chats/
    if LOCAL_CHATS_ROOT.exists():
        for c in LOCAL_CHATS_ROOT.rglob("_chat.txt"):
            if folder_source and folder_source in str(c):
                return c
    return None


def read_context_window(file_path: str, start_line: int, end_line: int,
                         pad: int = 10, folder_source: str = "") -> str:
    """Pull lines [start_line-pad, end_line+pad] from the source .txt."""
    p = _resolve_chat_file(file_path, folder_source)
    if p is None:
        return (f"[context unavailable: could not locate source file. "
                f"Tried `{file_path}` and `chats/{folder_source}/_chat.txt`.]")
    lo = max(1, start_line - pad)
    hi = end_line + pad
    out = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            if idx < lo:
                continue
            if idx > hi:
                break
            marker = ">>" if start_line <= idx <= end_line else "  "
            out.append(f"{marker} {idx:>5}: {line.rstrip()}")
    return "\n".join(out) or "[context unavailable: line range empty]"


def hits_to_csv(hits, ai_summary: str | None) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "chat_name", "account_owner", "start_date", "end_date",
        "start_line", "end_line", "senders",
        "has_voice", "has_call", "has_media", "distance", "chunk_text",
    ])
    for h in hits:
        m = h["meta"]
        w.writerow([
            m.get("chat_name", ""),
            _account_owner_from_file(m.get("file", "")),
            m.get("start_date", ""),
            m.get("end_date", ""),
            m.get("start_line", ""),
            m.get("end_line", ""),
            m.get("senders", ""),
            m.get("has_voice", ""),
            m.get("has_call", ""),
            m.get("has_media", ""),
            f"{h.get('distance', 0):.4f}",
            h["doc"],
        ])
    if ai_summary:
        w.writerow([])
        w.writerow(["AI_SUMMARY"])
        w.writerow([ai_summary])
    return buf.getvalue().encode("utf-8")


def hits_to_pdf(question: str, hits, ai_summary: str | None) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted,
        )
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    mono = ParagraphStyle("mono", parent=styles["Code"], fontSize=8, leading=10)
    story = [
        Paragraph("WhatsApp Forensic Audit Report", styles["Title"]),
        Paragraph(f"Query: <b>{question}</b>", styles["Normal"]),
        Paragraph(f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                  styles["Normal"]),
        Paragraph(f"Chunks: {len(hits)}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]
    if ai_summary:
        story.append(Paragraph("AI Audit Summary", styles["Heading2"]))
        for line in ai_summary.splitlines():
            story.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;")
                                   or "&nbsp;", styles["BodyText"]))
        story.append(PageBreak())

    story.append(Paragraph("Evidence Chunks (chronological)", styles["Heading2"]))
    for h in hits:
        m = h["meta"]
        head = (f"<b>{m.get('chat_name','')}</b> &middot; "
                f"{m.get('start_date','')} &rarr; {m.get('end_date','')} &middot; "
                f"lines {m.get('start_line','')}-{m.get('end_line','')}")
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph(head, styles["Normal"]))
        story.append(Preformatted(h["doc"], mono))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WhatsApp Forensic Audit",
    page_icon="🔎",
    layout="wide",
)

st.title("WhatsApp Forensic Audit Dashboard")
st.caption(f"Local vector DB: `{DB_PATH}` — collection: `{COLLECTION}`")

# DB sanity check ------------------------------------------------------------
try:
    facets = load_facets()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not open ChromaDB at `{DB_PATH}`.\n\n{e}")
    st.info("Make sure you've run `python ingest.py` and the "
            "`whatsapp_vector_db/` folder sits next to this app.")
    st.stop()

st.success(
    f"Index loaded: {facets['total']:,} chunks across {len(facets['chats'])} chats "
    f"({facets['earliest']} → {facets['latest']})."
)

# Sidebar --------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    sel_owners = st.multiselect(
        "Account owner (folder containing the chat)",
        facets["owners"],
        default=[],
        help="Derived from the folder one level above each chat folder.",
    )
    sel_chats = st.multiselect(
        "Chat",
        facets["chats"],
        default=[],
    )

    st.subheader("Evidence type")
    f_voice = st.checkbox("Voice notes / audio", value=False)
    f_call = st.checkbox("Call events", value=False)
    f_media = st.checkbox("Other media (image/video/doc)", value=False)

    st.subheader("Timeline")
    date_range = st.slider(
        "Date range",
        min_value=facets["earliest"],
        max_value=facets["latest"],
        value=(facets["earliest"], facets["latest"]),
        format="YYYY-MM-DD",
    )

    st.subheader("Retrieval")
    n_chunks = st.slider("Chunks to retrieve", 10, 200, 40, step=10)
    model = st.selectbox("LLM model", ["gpt-4o", "gpt-4o-mini"], index=0)

# Main panel -----------------------------------------------------------------
col_q, col_mode = st.columns([4, 1])
with col_q:
    question = st.text_area(
        "Investigation query",
        placeholder='e.g. "every time Santigie refused to pay"',
        height=90,
    )
with col_mode:
    mode = st.radio(
        "Output mode",
        ["AI Audit Summary", "Raw Evidence Dump"],
        index=0,
    )

run = st.button("Run audit", type="primary", use_container_width=True)

# Persist last-run state across reruns so download buttons survive.
if "last" not in st.session_state:
    st.session_state.last = None

if run:
    if not question.strip():
        st.warning("Enter a query first.")
        st.stop()

    with st.spinner("Searching ChromaDB..."):
        try:
            hits = run_query(
                question.strip(),
                n=n_chunks,
                chat_names=sel_chats,
                owner_names=sel_owners,
                has_voice=f_voice,
                has_call=f_call,
                has_media=f_media,
                start_d=date_range[0],
                end_d=date_range[1],
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

    ai_summary = None
    if mode == "AI Audit Summary":
        if not hits:
            st.warning("No chunks matched — skipping LLM.")
        else:
            try:
                with st.spinner(f"Calling {model}..."):
                    ai_summary = ask.call_llm(question.strip(), hits, model=model)
            except SystemExit as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"LLM call failed: {e}")

    st.session_state.last = {
        "question": question.strip(),
        "hits": hits,
        "summary": ai_summary,
        "mode": mode,
    }

# Render last run ------------------------------------------------------------
last = st.session_state.last
if last:
    hits = last["hits"]
    ai_summary = last["summary"]
    st.divider()
    st.subheader(f"Results — {len(hits)} chunks")

    if ai_summary:
        with st.container(border=True):
            st.markdown("### AI Audit Summary")
            st.markdown(ai_summary)

    if not hits:
        st.info("No chunks matched the filters.")
    else:
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            tags = []
            if m.get("has_voice") == "1":
                tags.append("🎙 voice")
            if m.get("has_call") == "1":
                tags.append("📞 call")
            if m.get("has_media") == "1":
                tags.append("🖼 media")
            tag_str = "  ".join(tags)
            owner = _account_owner_from_file(m.get("file", ""))
            header = (
                f"**{i}. {m.get('chat_name','(unknown)')}**  ·  "
                f"`{owner}`  ·  "
                f"{m.get('start_date','')} → {m.get('end_date','')}  ·  "
                f"lines {m.get('start_line','')}-{m.get('end_line','')}  "
                f"{tag_str}"
            )
            with st.container(border=True):
                st.markdown(header)
                st.code(h["doc"], language="text")
                with st.expander("Show ±10 lines of surrounding context from source file"):
                    ctx = read_context_window(
                        m.get("file", ""),
                        int(m.get("start_line", 1) or 1),
                        int(m.get("end_line", 1) or 1),
                        pad=10,
                        folder_source=m.get("folder_source", ""),
                    )
                    st.code(ctx, language="text")
                st.caption(f"distance={h.get('distance', 0):.3f}  ·  "
                           f"source: `{m.get('file','')}`")

    # Export ----------------------------------------------------------------
    st.divider()
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    with c1:
        st.download_button(
            "Download CSV",
            data=hits_to_csv(hits, ai_summary),
            file_name=f"whatsapp-audit-{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        pdf_bytes = hits_to_pdf(last["question"], hits, ai_summary)
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"whatsapp-audit-{ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button("Download PDF", disabled=True, use_container_width=True,
                      help="`pip install reportlab` to enable PDF export.")
    with c3:
        md = []
        md.append(f"# WhatsApp Forensic Audit\n\n**Query:** {last['question']}\n")
        if ai_summary:
            md.append("## AI Audit Summary\n\n" + ai_summary + "\n")
        md.append("## Evidence chunks\n")
        for h in hits:
            m = h["meta"]
            md.append(
                f"### {m.get('chat_name','')} — "
                f"{m.get('start_date','')} → {m.get('end_date','')} "
                f"(lines {m.get('start_line','')}-{m.get('end_line','')})\n\n"
                f"```\n{h['doc']}\n```\n"
            )
        st.download_button(
            "Download Markdown",
            data="\n".join(md).encode("utf-8"),
            file_name=f"whatsapp-audit-{ts}.md",
            mime="text/markdown",
            use_container_width=True,
        )

else:
    st.info("Set your filters in the sidebar, type a query above, and click "
            "**Run audit**.")
