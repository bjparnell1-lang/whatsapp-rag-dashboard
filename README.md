# WhatsApp Transcript RAG

Local ChromaDB vector index over every `.txt` chat export under `Whats App/`
(including `Whats App/bj/`). Optimized for forensic / timeline-style retrieval:
voice notes, call events, and media placeholders are preserved as evidence
markers, not ignored.

## Files

| File          | Purpose |
|---------------|---------|
| `ingest.py`   | Crawls .txt files, parses lines with regex, builds 8-line sliding-window chunks (stride 4), persists to ChromaDB. |
| `ask.py`      | Retrieves up to N chunks for a query, sorts chronologically, hands to GPT-4o as an objective auditor with `[Sender, Date]` citations. |
| `embedding.py`| Local TF-IDF + TruncatedSVD embedder (no network), with optional OpenAI fallback. |
| `whatsapp_vector_db/` | Persistent ChromaDB store + fitted vectorizer pickle. (Created by `ingest.py`.) |

## One-time setup

```bash
pip install chromadb openai langchain-text-splitters scikit-learn
```

## Build the index

If the chats are still in `.zip` form (WhatsApp exports), unzip the `_chat.txt`
files first; ingest only looks at `.txt`. From the `rag/` folder:

```bash
python ingest.py --reset
```

Defaults:
- root  = parent folder (the `Whats App/` directory)
- db    = `./whatsapp_vector_db`
- window=8, stride=4
- embedder = local TF-IDF+SVD (no network)

Override with `--root`, `--db`, `--window`, `--stride`, `--embedder openai`.

## Query

```bash
# Set the OpenAI key for the GPT-4o audit
export OPENAI_API_KEY=sk-...

# Compile every instance, chronologically, with citations
python ask.py "every time Santigie discussed payroll"

# Filter to one chat
python ask.py "voice notes about Dr. Arkhurst" --chat "BK MANAGEMENT"

# Only chunks that contain a voice note
python ask.py "messages right before Bernadette's voice notes" --has-voice -n 50

# Skip the LLM and dump raw retrieved chunks (no API key needed)
python ask.py "any payment refusal" --no-llm -n 30
```

## Chunk metadata captured per entry

- `folder_source`, `chat_name`, `file`
- `senders` (pipe-joined unique senders in the window)
- `start_date`, `end_date` (ISO 8601)
- `start_line`, `end_line` (line numbers in source .txt)
- `has_voice`, `has_call`, `has_media` (string "0"/"1" flags)

## Notes

- WhatsApp iOS export format: `[M/D/YY, H:MM:SS AM/PM] Sender: Message`.
  The parser strips the leading LRM/RLM Unicode marks and supports multi-line
  message bodies.
- Voice-note placeholders captured: `audio omitted`, `voice message omitted`,
  `voice note`, `<attached: ...AUDIO...opus>`, and `.opus` references.
- Call events captured: `voice call`, `video call`, `missed voice call/video call`.
- TF-IDF+SVD was chosen over a remote embedding model because (a) transcript
  evidence retrieval rewards exact-term matching, (b) it works offline, and
  (c) it builds in ~3s for the current corpus.
