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

# Hoist Streamlit Cloud secrets into os.environ.
try:
    for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = st.secrets[k]
except Exception:
    pass

# --- LLM config (OpenRouter, OpenAI-compatible) ----------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-30b-a3b:free"
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-r1:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

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


CATEGORIES = ["Robert", "Kharka", "Group"]


def _is_robert_path(file_path: str) -> bool:
    """True if the chat sits inside the Bj/ subfolder (Robert's exports)."""
    fp = (file_path or "").replace("\\", "/").lower()
    return "/bj/" in fp


def classify_chat(chat_name: str, file_path: str, unique_sender_count: int) -> str:
    """
    Robert -> file lives under Bj/ (Robert's WhatsApp exports)
    Group  -> chat has more than 2 unique senders (group chat)
    Kharka -> everything else (1-on-1 chats from the main account)
    """
    if _is_robert_path(file_path):
        return "Robert"
    if unique_sender_count > 2:
        return "Group"
    return "Kharka"


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
    """Walk metadata to compute facets for the sidebar.
    Aggregates unique senders per chat so we can classify each chat
    as Robert / Kharka / Group exactly once (not per-chunk).
    """
    col = open_collection()
    total = col.count()
    chats = set()
    chat_senders: dict[str, set[str]] = {}   # chat_name -> set of senders
    chat_file: dict[str, str] = {}           # chat_name -> a representative file path
    earliest, latest = None, None
    BATCH = 1000
    fetched = 0
    while fetched < total:
        res = col.get(limit=BATCH, offset=fetched, include=["metadatas"])
        metas = res.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            cname = m.get("chat_name", "")
            chats.add(cname)
            chat_file.setdefault(cname, m.get("file", ""))
            for s in (m.get("senders", "") or "").split(" | "):
                s = s.strip()
                if s:
                    chat_senders.setdefault(cname, set()).add(s)
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

    # Classify each chat into a category.
    chat_category: dict[str, str] = {}
    for c in chats:
        chat_category[c] = classify_chat(
            c, chat_file.get(c, ""), len(chat_senders.get(c, set()))
        )

    return {
        "chats": sorted(c for c in chats if c),
        "chat_category": chat_category,
        "chat_senders": {c: sorted(s) for c, s in chat_senders.items()},
        "earliest": earliest or date(2020, 1, 1),
        "latest": latest or date.today(),
        "total": total,
    }


def run_query(question, n, chat_names, categories, has_voice, has_call, has_media,
              start_d, end_d, chat_category_map):
    """Run ask.query() then apply post-hoc filters (chat/category/date/has_media)."""
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
    cats_set = set(categories or [])

    def keep(h):
        m = h["meta"]
        cname = m.get("chat_name", "")
        if chats_set and cname not in chats_set:
            return False
        if cats_set and chat_category_map.get(cname, "Kharka") not in cats_set:
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


def call_llm(question, hits, model):
    """LLM call via OpenRouter (OpenAI-compatible API).
    Reuses ask.SYSTEM_PROMPT and ask.dump_context so audit behavior is unchanged.
    """
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. In Streamlit Cloud, open "
            "Settings → Secrets and add:\n\n"
            '    OPENROUTER_API_KEY = "sk-or-v1-..."'
        )
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        # OpenRouter uses these for free-tier rate limiting and the model gallery.
        default_headers={
            "HTTP-Referer": "https://whatsappbk.streamlit.app",
            "X-Title": "WhatsApp Forensic Audit",
        },
    )
    user_msg = (
        f"USER QUESTION:\n{question}\n\n"
        f"EXCERPTS (already pre-sorted chronologically, {len(hits)} chunks):\n\n"
        + ask.dump_context(hits)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ask.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


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


def hits_to_csv(hits, ai_summary: str | None, chat_category_map: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "chat_name", "category", "start_date", "end_date",
        "start_line", "end_line", "senders",
        "has_voice", "has_call", "has_media", "distance", "chunk_text",
    ])
    for h in hits:
        m = h["meta"]
        w.writerow([
            m.get("chat_name", ""),
            chat_category_map.get(m.get("chat_name", ""), ""),
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

    sel_categories = st.multiselect(
        "Account / category",
        CATEGORIES,
        default=[],
        help=(
            "**Robert** = chats from the `Bj/` subfolder (your personal exports). "
            "**Kharka** = 1-on-1 chats from the main account. "
            "**Group** = chats with more than 2 participants."
        ),
    )

    # Chat options narrow down to whatever categories are selected.
    if sel_categories:
        chat_options = [
            c for c in facets["chats"]
            if facets["chat_category"].get(c) in set(sel_categories)
        ]
    else:
        chat_options = facets["chats"]

    sel_chats = st.multiselect(
        "Specific chats (optional)",
        chat_options,
        default=[],
        help="Narrow further to specific chat threads within the chosen categories.",
    )

    # Show a small breakdown so you can sanity-check the classification.
    with st.expander("How chats are categorized", expanded=False):
        for cat in CATEGORIES:
            members = [c for c, v in facets["chat_category"].items() if v == cat]
            st.markdown(f"**{cat}** ({len(members)})")
            for c in sorted(members):
                st.markdown(f"- {c}")

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

    st.subheader("LLM (via OpenRouter)")
    model = st.selectbox(
        "Model",
        FALLBACK_MODELS,
        index=0,
        help=(
            "Free OpenRouter models. If one is rate-limited or unavailable, "
            "try another. Requires `OPENROUTER_API_KEY` in Streamlit Secrets."
        ),
    )
    custom_model = st.text_input(
        "Or paste a custom model ID",
        value="",
        placeholder="e.g. anthropic/claude-3.5-sonnet",
        help="Overrides the dropdown. Leave empty to use the dropdown selection.",
    )
    if custom_model.strip():
        model = custom_model.strip()

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
                categories=sel_categories,
                has_voice=f_voice,
                has_call=f_call,
                has_media=f_media,
                start_d=date_range[0],
                end_d=date_range[1],
                chat_category_map=facets["chat_category"],
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
                with st.spinner(f"Calling {model} via OpenRouter..."):
                    ai_summary = call_llm(question.strip(), hits, model=model)
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
            category = facets["chat_category"].get(m.get("chat_name", ""), "")
            header = (
                f"**{i}. {m.get('chat_name','(unknown)')}**  ·  "
                f"`{category}`  ·  "
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
            data=hits_to_csv(hits, ai_summary, facets["chat_category"]),
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
