"""
app.py
------
Streamlit UI for the local PDF RAG chatbot.

Now backed by a persistent ChromaDB vector store: documents you process
stay indexed across app restarts (stored in ./chroma_db on disk), you can
browse what's actually stored, and remove individual documents.

Run with:  streamlit run app.py

Requires Ollama running locally with:
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ollama pull llava                # only needed for image analysis (scanned pages/figures)
"""

import os
import tempfile
import time

import streamlit as st
import pandas as pd

from pdf_processor import extract_pages
from chunking import RecursiveCharacterTextSplitter
from vector_store import VectorStore, DEFAULT_PERSIST_DIR
from reranker import Reranker
from rag_pipeline import RAGPipeline
from image_processor import extract_scanned_pages, extract_embedded_figures, extract_vector_figures
from analytics import log_query, load_query_log, new_query_entry, document_stats, pca_2d

st.set_page_config(page_title="Local PDF RAG Chatbot", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------
# Theme: "reading room, after hours" — this is a local, private tool for
# digging through your own documents, so the palette leans into that:
# a dark ink background like a desk lamp left on late, one warm amber
# accent carrying all emphasis (buttons, headers, active states), and a
# muted teal reserved only for citations/page references so it reads as
# an archival index mark rather than competing for attention.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink-900: #12141C;
        --ink-800: #1A1D2A;
        --ink-700: #242838;
        --amber: #E8A33D;
        --amber-light: #F0C168;
        --parchment: #ECE4D6;
        --cite-teal: #4FA894;
    }

    /* Sidebar reads as a shelf beside the main reading desk */
    section[data-testid="stSidebar"] {
        background-color: var(--ink-800);
        border-right: 1px solid var(--ink-700);
    }

    /* Page title stays plain — the page icon carries the visual anchor
       instead of a decorative accent bar. Section headers (h2, h3) get a
       subtle neutral rule so nothing competes with the title. */
    h1 {
        font-weight: 700;
        border-bottom: none;
    }
    h2, h3 {
        border-bottom: 1px solid var(--ink-700);
        padding-bottom: 0.3rem;
        font-weight: 500;
        color: var(--parchment);
        opacity: 0.92;
    }

    /* Buttons: amber fill, ink text, no shadow — the one bold element */
    .stButton > button, .stDownloadButton > button {
        background-color: var(--amber);
        color: var(--ink-900);
        border: none;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: var(--amber-light);
        color: var(--ink-900);
    }

    /* Chat messages get a bookmark-ribbon accent instead of a boxed card */
    div[data-testid="stChatMessage"] {
        border-left: 3px solid var(--amber);
        padding-left: 0.9rem;
        background-color: transparent;
    }

    /* Page/citation references (backtick code spans) read as index-card marks */
    code {
        color: var(--cite-teal);
        background-color: rgba(79, 168, 148, 0.12);
    }

    /* Metrics on the Dashboard tab pick up the amber accent for the number */
    div[data-testid="stMetricValue"] {
        color: var(--amber);
    }

    /* Slim, deliberate dividers rather than heavy default rules */
    hr {
        border-top: 1px solid var(--ink-700);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Cached, expensive-to-create objects.
#
# NOTE on embed_model: the persisted collection is tied to whichever
# embedding model was used to build it. If you change the embedding model
# in the sidebar after documents are already indexed, new chunks will be
# embedded with the new model while old ones stay embedded with the old
# one — similarity scores between the two become meaningless. Stick to one
# embedding model per chroma_db folder, or clear the index before switching.
# ---------------------------------------------------------------------


@st.cache_resource
def load_reranker():
    return Reranker()


@st.cache_resource
def get_vector_store(embed_model: str, persist_dir: str = DEFAULT_PERSIST_DIR):
    # cached so the same ChromaDB client/collection is reused across
    # Streamlit reruns instead of reopening the persisted folder every time
    return VectorStore(embed_model=embed_model, persist_dir=persist_dir)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session_state()

# ---------------------------------------------------------------------
# Sidebar: settings, upload + process PDFs, browse/manage the index
# ---------------------------------------------------------------------

with st.sidebar:
    st.subheader("Settings")
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, step=25)
    retrieve_k = st.slider("Chunks retrieved (bi-encoder)", 5, 40, 20)
    rerank_k = st.slider("Chunks kept after rerank", 1, 10, 5)
    llm_model = st.text_input("Ollama LLM model", value="llama3.2")
    embed_model = st.text_input("Ollama embedding model", value="nomic-embed-text")
    corrective_mode = st.checkbox(
        "Enable self-checking (corrective RAG)",
        value=False,
        help="After generating an answer, the model grades whether it's "
        "actually supported by the retrieved context. If not, it rewrites "
        "the question and retries once. More accurate, but slower and "
        "non-streaming (grading needs the full answer first).",
    )
    analyze_images = st.checkbox(
        "Analyze images/scanned pages",
        value=False,
        help="Runs a vision model over scanned pages, embedded figures, and "
        "vector-drawn charts to make them searchable too. Off by default "
        "because most PDFs are text-only, and vision analysis is by far "
        "the slowest part of processing — every page gets checked even "
        "when there's nothing for it to find. Turn this on only for PDFs "
        "you know contain scanned content or charts you want to ask about.",
    )

    VISION_MODEL = "llava"

    # This is created/reused as soon as we know which embedding model to use,
    # so persisted data from previous sessions is visible immediately below.
    store = get_vector_store(embed_model=embed_model)

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDF document(s)", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        with st.status("Processing PDFs...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                st.write(f"Parsing **{uploaded_file.name}**...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                pages = extract_pages(tmp_path, source_name=uploaded_file.name)
                st.write(f"→ {len(pages)} text pages extracted")

                # Always runs — this extracts plain text via PyMuPDF, which
                # is fast even on large documents. Vision-model analysis
                # (scanned pages, figures, charts) is gated behind the
                # "Analyze images/scanned pages" checkbox above, since
                # that's the slow part.
                # Only runs when the "Analyze images/scanned pages" checkbox
                # is on — this is the slowest part of processing (a vision
                # model call per candidate page/image), so it's opt-in
                # rather than running unconditionally on every upload.
                if analyze_images:
                    st.write("Checking for scanned pages (vision model)...")
                    scanned_pages = extract_scanned_pages(
                        tmp_path, source_name=uploaded_file.name, model=VISION_MODEL
                    )
                    if scanned_pages:
                        pages += scanned_pages
                        st.write(f"→ {len(scanned_pages)} scanned pages transcribed")

                    st.write("Checking for embedded figures (vision model)...")
                    figure_pages = extract_embedded_figures(
                        tmp_path, source_name=uploaded_file.name, model=VISION_MODEL
                    )
                    if figure_pages:
                        pages += figure_pages
                        st.write(f"→ {len(figure_pages)} embedded figures described")

                    st.write("Checking for vector-drawn charts/diagrams (vision model)...")
                    vector_figure_pages = extract_vector_figures(
                        tmp_path, source_name=uploaded_file.name, model=VISION_MODEL
                    )
                    if vector_figure_pages:
                        pages += vector_figure_pages
                        st.write(f"→ {len(vector_figure_pages)} vector diagrams described")
                else:
                    scanned_pages, figure_pages, vector_figure_pages = [], [], []

                if not scanned_pages and not figure_pages and not vector_figure_pages and not pages:
                    st.warning(
                        f"No text or images could be extracted from {uploaded_file.name}. "
                        "It may be empty, corrupted, or the vision model may be unavailable "
                        "(check that VISION_MODEL is pulled: `ollama pull llava`)."
                        if analyze_images
                        else f"No text could be extracted from {uploaded_file.name}. "
                        "It may be a scanned/image-only PDF — try turning on "
                        "'Analyze images/scanned pages' above."
                    )

                os.unlink(tmp_path)

                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                st.write(f"→ {len(chunks)} chunks total for this file")

            progress_bar = st.progress(0.0, text="Embedding chunks...")

            def progress_callback(done, total):
                progress_bar.progress(done / total, text=f"Embedding & storing... {done}/{total}")

            store.add_chunks(all_chunks, progress_callback=progress_callback)
            status.update(label="Done!", state="complete")

        st.success(f"Indexed {len(all_chunks)} chunks. Total in database: {len(store)}.")

    # ---- Manage what's currently indexed (persists across restarts) ----
    sources = store.list_sources()
    if sources:
        st.divider()
        st.caption(f"Indexed files ({len(store)} chunks total):")
        for fname in sources:
            col1, col2 = st.columns([4, 1])
            col1.text(f"• {fname}")
            if col2.button("🗑️", key=f"del_{fname}", help=f"Remove {fname} from the index"):
                store.delete_source(fname)
                st.rerun()

        with st.expander("🔍 Browse indexed chunks"):
            st.caption("A sample of what's actually stored in the database, including from previous sessions.")
            for item in store.peek(limit=10):
                meta = item["metadata"]
                st.markdown(f"**{meta.get('source')}** — `page {meta.get('page_number')}`")
                st.text(item["text"][:300] + ("..." if len(item["text"]) > 300 else ""))
                st.divider()

        if st.button("Clear entire index"):
            store.clear()
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------
# Main area: Chat and Dashboard tabs
# ---------------------------------------------------------------------

if len(store) == 0:
    st.info("Upload one or more PDFs in the sidebar and click **Process documents** to get started.")
    st.stop()

st.title("📄 Local PDF RAG")
st.caption(f"{len(store.list_sources())} document(s) indexed, {len(store)} chunks total")

chat_tab, dashboard_tab = st.tabs(["💬 Chat", "📊 Dashboard"])

with chat_tab:
    all_sources = store.list_sources()
    selected_sources = st.multiselect(
        "Ask about",
        options=all_sources,
        default=all_sources,
        help="Only chunks from the selected document(s) will be searched. "
        "Leave all selected to ask across every uploaded PDF; narrow it "
        "down to stop answers from one PDF leaking into questions about another.",
    )
    # Treat "everything selected" the same as "no filter" — this also
    # covers newly-uploaded files that weren't in `all_sources` at the
    # moment the widget was drawn.
    active_filter = None if set(selected_sources) == set(all_sources) else selected_sources
    st.session_state["selected_sources"] = selected_sources
    st.session_state["all_sources"] = all_sources

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources used"):
                    for src in msg["sources"]:
                        meta = src["metadata"]
                        st.markdown(
                            f"**{meta.get('source')}** — `page {meta.get('page_number')}` "
                            f"(rerank score: {src.get('rerank_score', 0):.3f})"
                        )
                        st.text(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))

    if not selected_sources:
        st.warning("Select at least one document above to ask a question.")
    question = st.chat_input(
        "Ask a question about your document(s)...", disabled=not selected_sources
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        reranker = load_reranker()
        pipeline = RAGPipeline(
            vector_store=store,
            reranker=reranker,
            llm_model=llm_model,
            retrieve_k=retrieve_k,
            rerank_k=rerank_k,
        )

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:-1]
        ]

        start_time = time.time()
        supported_flag = None

        with st.chat_message("assistant"):
            if corrective_mode:
                # Self-checking path: needs the full answer before it can
                # grade it, so this isn't streamed — the tradeoff for the
                # accuracy check.
                with st.spinner("Retrieving, generating, and self-checking the answer..."):
                    answer, sources, attempts = pipeline.answer_with_correction(
                        question, chat_history=history, source_filter=active_filter
                    )
                supported_flag = attempts[-1]["supported"] if attempts else None
                st.markdown(answer)
                if len(attempts) > 1:
                    st.caption(
                        f"⚠️ First answer was flagged as unsupported by the retrieved "
                        f"context — retried with a rewritten query: "
                        f"\"{attempts[1]['query_used']}\""
                    )
            else:
                with st.spinner("Retrieving relevant chunks..."):
                    token_stream, sources = pipeline.answer_stream(
                        question, chat_history=history, source_filter=active_filter
                    )
                answer = st.write_stream(token_stream)

            with st.expander("Sources used"):
                for src in sources:
                    meta = src["metadata"]
                    st.markdown(
                        f"**{meta.get('source')}** — `page {meta.get('page_number')}` "
                        f"(rerank score: {src.get('rerank_score', 0):.3f})"
                    )
                    st.text(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))

        latency = time.time() - start_time
        log_query(new_query_entry(question, latency, len(sources), supported_flag))

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with dashboard_tab:
    dash_selected = st.session_state.get("selected_sources", store.list_sources())
    dash_all = st.session_state.get("all_sources", store.list_sources())
    dash_scoped = bool(dash_selected) and set(dash_selected) != set(dash_all)

    st.subheader("Document overview")
    if dash_scoped:
        st.caption(
            f"Showing analysis for {len(dash_selected)} of {len(dash_all)} document(s) — "
            "matches your \"Ask about\" selection in the Chat tab."
        )
    raw_data = store.get_all()
    if dash_scoped:
        keep_idx = [i for i, m in enumerate(raw_data["metadatas"]) if m.get("source") in dash_selected]
        all_data = {
            "ids": [raw_data["ids"][i] for i in keep_idx],
            "embeddings": [raw_data["embeddings"][i] for i in keep_idx],
            "metadatas": [raw_data["metadatas"][i] for i in keep_idx],
            "documents": [raw_data["documents"][i] for i in keep_idx],
        }
    else:
        all_data = raw_data
    stats = document_stats(all_data["metadatas"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(stats))
    col2.metric("Total chunks", len(all_data["ids"]))
    col3.metric("Total pages", sum(v["pages"] for v in stats.values()))

    if stats:
        df_docs = pd.DataFrame(
            [{"Document": src, "Chunks": v["chunks"], "Pages": v["pages"]} for src, v in stats.items()]
        )
        st.dataframe(df_docs, use_container_width=True, hide_index=True)

    st.subheader("Embedding space (2D projection)")
    st.caption(
        "Each point is one chunk, projected from its full embedding down to 2D via PCA. "
        "Points that cluster together are semantically similar — useful for sanity-checking "
        "whether your chunking is producing coherent, well-separated topics."
    )
    if len(all_data["ids"]) >= 2:
        coords = pca_2d(all_data["embeddings"])
        df_embed = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "source": [m.get("source", "unknown") for m in all_data["metadatas"]],
            }
        )
        st.scatter_chart(df_embed, x="x", y="y", color="source")
    else:
        st.caption("Need at least 2 chunks indexed to show this.")

    st.subheader("Query history")
    log = load_query_log()
    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(
            df_log[["timestamp", "question", "latency_sec", "num_sources", "supported"]],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Latency over time (seconds)**")
        st.line_chart(df_log.set_index("timestamp")["latency_sec"])
    else:
        st.caption("No queries logged yet — ask something in the Chat tab.")