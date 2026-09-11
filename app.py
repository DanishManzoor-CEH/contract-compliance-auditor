"""
Smart Multi-Document Compliance & Contract Auditor
====================================================

A Streamlit application that lets a user upload one or more PDF contracts /
policy documents, builds a FAISS vector index over them, and answers
compliance questions using a Corrective RAG pipeline:

    Extract -> Chunk -> Embed -> Retrieve -> LLM Answer
        -> Self-Verification / Groundedness Check -> UI Output

LLM: Groq (llama-3.3-70b-versatile / llama3-8b-8192) via langchain-groq
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, local, free)
Vector store: FAISS (CPU, in-memory, cached in st.session_state)

Author: Generated for the "contract-compliance-auditor" project.
"""

import os
import io
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# HuggingFaceEmbeddings moved from langchain_community to langchain_huggingface.
# We try the new package first and gracefully fall back to keep the app
# working regardless of which version the user has installed.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------------------------------
# 0. ENVIRONMENT & PAGE CONFIGURATION
# --------------------------------------------------------------------------

load_dotenv()  # allows a local .env file to populate os.environ

st.set_page_config(
    page_title="Smart Multi-Document Compliance & Contract Auditor",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ---- Global font & background tweaks ---- */
    .main {
        background-color: #0f172a;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---- Title styling ---- */
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* ---- Cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    /* ---- Verdict badges ---- */
    .badge-grounded {
        display: inline-block;
        background-color: #14532d;
        color: #bbf7d0;
        border: 1px solid #22c55e;
        border-radius: 999px;
        padding: 0.15rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-warning {
        display: inline-block;
        background-color: #451a03;
        color: #fde68a;
        border: 1px solid #f59e0b;
        border-radius: 999px;
        padding: 0.15rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-unverified {
        display: inline-block;
        background-color: #450a0a;
        color: #fecaca;
        border: 1px solid #ef4444;
        border-radius: 999px;
        padding: 0.15rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ---- Source chunk box ---- */
    .source-chunk {
        background-color: #111827;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        color: #e2e8f0;
        white-space: pre-wrap;
    }
    .source-meta {
        font-size: 0.78rem;
        color: #93c5fd;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    /* ---- Chat bubbles ---- */
    .stChatMessage {
        border-radius: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 1. CONSTANTS
# --------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_ANSWER_PROMPT = """You are an expert contract and compliance auditor AI.
You answer strictly using the CONTEXT provided below, which was retrieved from
the user's uploaded documents. You are meticulous, precise, and conservative:
you never invent clauses, dates, obligations, or numbers that are not present
in the context.

Rules:
1. Answer only using information found in the CONTEXT.
2. If the CONTEXT does not contain enough information to answer confidently,
   explicitly say so instead of guessing.
3. Every factual claim or clause you reference MUST be followed by an inline
   citation in the exact format: [Source: <filename>, Page <page_number>].
4. If multiple sources support a claim, cite all of them.
5. Structure your answer with clear markdown: use headers, bullet points, or
   numbered lists where it improves clarity (e.g., listing obligations,
   deadlines, penalties, or compliance gaps).
6. Keep the tone formal and objective, as befits a compliance report.

CONTEXT:
{context}

QUESTION:
{question}

Write your grounded, cited answer below:
"""

VERIFICATION_PROMPT = """You are a strict fact-checking auditor performing a
"groundedness verification" pass on a draft answer produced by another AI.

Your job: compare every statement in the DRAFT ANSWER against the SOURCE
CONTEXT chunks. Flag any statement that is not directly supported by the
context (a hallucination, an unsupported inference, or a missing citation).

SOURCE CONTEXT:
{context}

DRAFT ANSWER:
{draft_answer}

Respond in the following strict markdown format and nothing else:

VERDICT: <one of: FULLY_GROUNDED | PARTIALLY_GROUNDED | UNVERIFIED>

SUMMARY: <one or two sentence summary of your assessment>

ISSUES:
- <bullet list of any unsupported/ungrounded claims, or "None found." if
  the draft is fully grounded>

CORRECTED_ANSWER:
<If VERDICT is FULLY_GROUNDED, simply repeat the draft answer unchanged.
If PARTIALLY_GROUNDED or UNVERIFIED, rewrite the answer removing or
qualifying any ungrounded claims, keeping only statements that are directly
supported by the SOURCE CONTEXT, while preserving correct citations.>
"""


# --------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "vectorstore": None,
        "processed_files": [],
        "chat_history": [],  # list of dicts: {role, content, sources, verdict}
        "embeddings_model": None,
        "processing_done": False,
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --------------------------------------------------------------------------
# 3. DOCUMENT PROCESSING PIPELINE
# --------------------------------------------------------------------------

def extract_documents_from_pdfs(uploaded_files: List[Any]) -> List[Document]:
    """
    Reads each uploaded PDF with pypdf, extracts text page-by-page, and
    returns a list of LangChain Document objects with metadata preserving
    the source filename and the 1-indexed page number.
    """
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            reader = PdfReader(io.BytesIO(file_bytes))

            for page_index, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()

                if not page_text:
                    # Skip empty / scanned-image pages with no extractable text
                    continue

                doc = Document(
                    page_content=page_text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_index + 1,  # human-readable, 1-indexed
                    },
                )
                all_docs.append(doc)

        except Exception as exc:
            st.error(f"❌ Failed to read '{uploaded_file.name}': {exc}")

    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits page-level documents into overlapping chunks while preserving
    the original source/page metadata on every resulting chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """
    Loads (and caches across reruns) the HuggingFace sentence-transformers
    embedding model. Cached with st.cache_resource so the model is only
    downloaded/initialized once per session/server process.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document], embeddings) -> FAISS:
    """
    Builds a FAISS vector store in memory from the provided chunks and
    embedding model.
    """
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


def process_uploaded_documents(uploaded_files: List[Any]) -> None:
    """
    Full pipeline orchestration: Extract -> Chunk -> Embed -> Index.
    Populates st.session_state with the resulting FAISS vector store.
    """
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file before processing.")
        return

    progress_bar = st.sidebar.progress(0, text="Starting document pipeline...")

    # Step 1: Extract
    progress_bar.progress(15, text="📄 Extracting text from PDFs...")
    raw_docs = extract_documents_from_pdfs(uploaded_files)

    if not raw_docs:
        progress_bar.empty()
        st.error(
            "❌ No extractable text found in the uploaded PDF(s). "
            "They may be scanned images without an OCR text layer."
        )
        return

    # Step 2: Chunk
    progress_bar.progress(40, text="✂️ Splitting into overlapping chunks...")
    chunks = chunk_documents(raw_docs)

    # Step 3: Embed
    progress_bar.progress(65, text="🧠 Loading embedding model (MiniLM-L6-v2)...")
    embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)
    st.session_state.embeddings_model = embeddings

    # Step 4: Index
    progress_bar.progress(85, text="📚 Building FAISS vector index...")
    vectorstore = build_vector_store(chunks, embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.processed_files = [f.name for f in uploaded_files]
    st.session_state.total_chunks = len(chunks)
    st.session_state.processing_done = True

    progress_bar.progress(100, text="✅ Indexing complete!")
    time.sleep(0.4)
    progress_bar.empty()
    st.sidebar.success(
        f"✅ Indexed {len(uploaded_files)} document(s) into {len(chunks)} chunks."
    )


# --------------------------------------------------------------------------
# 4. RAG RETRIEVAL + LLM ANSWER + CORRECTIVE VERIFICATION
# --------------------------------------------------------------------------

def get_groq_llm(api_key: str, model_name: str, temperature: float = 0.1) -> ChatGroq:
    """
    Instantiates a ChatGroq LLM client with the given API key and model.
    Low temperature is used deliberately since this is a compliance/audit
    use case where determinism and faithfulness matter more than creativity.
    """
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=2048,
    )


def format_context(source_docs: List[Document]) -> str:
    """
    Formats retrieved chunks into a single context string, each block
    clearly tagged with its filename and page number so the LLM can cite
    them accurately.
    """
    blocks = []
    for i, doc in enumerate(source_docs, start=1):
        src = doc.metadata.get("source", "unknown_file")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[Chunk {i} | Source: {src} | Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def retrieve_relevant_chunks(query: str, k: int = RETRIEVAL_K) -> List[Document]:
    """
    Runs a similarity search against the FAISS index for the given query.
    """
    vectorstore: FAISS = st.session_state.vectorstore
    return vectorstore.similarity_search(query, k=k)


def generate_draft_answer(llm: ChatGroq, query: str, context: str) -> str:
    """
    Stage 1 of Corrective RAG: generate an initial cited answer grounded
    in the retrieved context.
    """
    prompt = SYSTEM_ANSWER_PROMPT.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content.strip()


def run_groundedness_verification(
    llm: ChatGroq, context: str, draft_answer: str
) -> Dict[str, str]:
    """
    Stage 2 of Corrective RAG: the "Self-Verification / Groundedness Check".
    Asks the LLM to critique its own draft answer against the source
    context, and returns a structured dict with verdict, summary, issues,
    and a corrected/final answer.
    """
    prompt = VERIFICATION_PROMPT.format(context=context, draft_answer=draft_answer)
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    parsed = {
        "verdict": "UNVERIFIED",
        "summary": "",
        "issues": "",
        "corrected_answer": draft_answer,
        "raw": raw_text,
    }

    try:
        # Simple, robust section-based parsing of the structured verification output.
        sections = {"VERDICT:": "verdict", "SUMMARY:": "summary",
                    "ISSUES:": "issues", "CORRECTED_ANSWER:": "corrected_answer"}

        # Find start indices of each marker present in the raw text.
        markers = []
        for marker in sections:
            idx = raw_text.find(marker)
            if idx != -1:
                markers.append((idx, marker))
        markers.sort()

        for i, (start_idx, marker) in enumerate(markers):
            content_start = start_idx + len(marker)
            content_end = markers[i + 1][0] if i + 1 < len(markers) else len(raw_text)
            value = raw_text[content_start:content_end].strip()
            key = sections[marker]
            if key == "verdict":
                # normalize to just the keyword
                for candidate in ["FULLY_GROUNDED", "PARTIALLY_GROUNDED", "UNVERIFIED"]:
                    if candidate in value:
                        value = candidate
                        break
            parsed[key] = value

        if not parsed["corrected_answer"]:
            parsed["corrected_answer"] = draft_answer

    except Exception:
        # If parsing fails for any reason, fall back gracefully to the raw
        # verification text and the original draft answer.
        parsed["summary"] = "Automatic parsing of verification output failed; showing raw output."
        parsed["issues"] = raw_text
        parsed["corrected_answer"] = draft_answer

    return parsed


def run_corrective_rag_pipeline(
    query: str, api_key: str, model_name: str
) -> Dict[str, Any]:
    """
    Orchestrates the full Corrective RAG flow for a single user query:
        Retrieve -> LLM Draft Answer -> Groundedness Verification -> Final Output
    """
    llm = get_groq_llm(api_key=api_key, model_name=model_name)

    source_docs = retrieve_relevant_chunks(query, k=RETRIEVAL_K)
    context = format_context(source_docs)

    draft_answer = generate_draft_answer(llm, query, context)
    verification = run_groundedness_verification(llm, context, draft_answer)

    return {
        "query": query,
        "draft_answer": draft_answer,
        "final_answer": verification["corrected_answer"],
        "verdict": verification["verdict"],
        "verification_summary": verification["summary"],
        "verification_issues": verification["issues"],
        "source_docs": source_docs,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# 5. UI RENDERING HELPERS
# --------------------------------------------------------------------------

def verdict_badge_html(verdict: str) -> str:
    verdict = (verdict or "").upper()
    if "FULLY" in verdict:
        return '<span class="badge-grounded">✅ FULLY GROUNDED</span>'
    elif "PARTIALLY" in verdict:
        return '<span class="badge-warning">⚠️ PARTIALLY GROUNDED</span>'
    else:
        return '<span class="badge-unverified">🚫 UNVERIFIED</span>'


def render_sources_expander(source_docs: List[Document]) -> None:
    with st.expander("📎 View Retrieved Source Chunks & Citations", expanded=False):
        if not source_docs:
            st.info("No source chunks were retrieved for this query.")
            return
        for i, doc in enumerate(source_docs, start=1):
            src = doc.metadata.get("source", "unknown_file")
            page = doc.metadata.get("page", "?")
            st.markdown(
                f'<div class="source-meta">Chunk {i} &nbsp;·&nbsp; '
                f'📄 {src} &nbsp;·&nbsp; Page {page}</div>'
                f'<div class="source-chunk">{doc.page_content}</div>',
                unsafe_allow_html=True,
            )


def render_verification_expander(result: Dict[str, Any]) -> None:
    with st.expander("🔍 View Corrective Verification Details", expanded=False):
        st.markdown(f"**Verdict:** {verdict_badge_html(result['verdict'])}", unsafe_allow_html=True)
        if result.get("verification_summary"):
            st.markdown(f"**Assessment Summary:** {result['verification_summary']}")
        if result.get("verification_issues"):
            st.markdown("**Flagged Issues:**")
            st.markdown(result["verification_issues"])
        st.markdown("---")
        st.markdown("**Original Draft Answer (pre-verification):**")
        st.markdown(result["draft_answer"])


# --------------------------------------------------------------------------
# 6. SIDEBAR — CONTROLS
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ---- API Key resolution: secrets.toml -> env var -> manual input ----
    default_api_key = ""
    try:
        default_api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_api_key = os.environ.get("GROQ_API_KEY", "")

    api_key_input = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        placeholder="gsk_...",
        help=(
            "If left blank, the app will fall back to GROQ_API_KEY from "
            "Streamlit secrets or your local .env file."
        ),
    )
    resolved_api_key = api_key_input.strip() or default_api_key

    if resolved_api_key:
        st.success("🔑 API key loaded.")
    else:
        st.warning("🔑 No Groq API key found. Enter one above to proceed.")

    st.markdown("---")

    selected_model = st.selectbox(
        "LLM Model (Groq)",
        options=AVAILABLE_MODELS,
        index=0,
        help="llama-3.3-70b-versatile is more accurate; llama3-8b-8192 is faster/cheaper.",
    )

    st.markdown("---")
    st.markdown("## 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF contracts / policy documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned images without OCR text will not extract.",
    )

    process_clicked = st.button("🚀 Process Documents", use_container_width=True, type="primary")

    if process_clicked:
        if not resolved_api_key:
            st.error("❌ Please provide a Groq API key before processing.")
        else:
            process_uploaded_documents(uploaded_files)

    if st.session_state.processing_done:
        st.markdown("---")
        st.markdown("## 📊 Index Status")
        st.markdown(
            f'<div class="metric-card">'
            f'<b>Files indexed:</b> {len(st.session_state.processed_files)}<br>'
            f'<b>Total chunks:</b> {st.session_state.total_chunks}<br>'
            f'<b>Embedding model:</b> MiniLM-L6-v2'
            f'</div>',
            unsafe_allow_html=True,
        )
        for fname in st.session_state.processed_files:
            st.caption(f"✔️ {fname}")

        if st.button("🗑️ Clear Index & Chat", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.session_state.processing_done = False
            st.session_state.total_chunks = 0
            st.rerun()


# --------------------------------------------------------------------------
# 7. MAIN PAGE — HEADER
# --------------------------------------------------------------------------

st.markdown(
    '<div class="app-title">📑 Smart Multi-Document Compliance & Contract Auditor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Multi-Stage Corrective RAG — Upload contracts, ask compliance '
    'questions, and get answers with verified citations and a self-grounded audit trail.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.processing_done:
    st.info(
        "👈 Start by uploading one or more PDF contracts in the sidebar, then click "
        "**Process Documents** to build the searchable index."
    )


# --------------------------------------------------------------------------
# 8. MAIN PAGE — CHAT HISTORY RENDER
# --------------------------------------------------------------------------

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(verdict_badge_html(turn["verdict"]), unsafe_allow_html=True)
        st.markdown(turn["final_answer"])
        render_sources_expander(turn["source_docs"])
        render_verification_expander(turn)


# --------------------------------------------------------------------------
# 9. MAIN PAGE — CHAT INPUT
# --------------------------------------------------------------------------

user_query = st.chat_input(
    "Ask a compliance question, e.g. 'What are the termination clauses and their notice periods?'"
)

if user_query:
    if not st.session_state.processing_done or st.session_state.vectorstore is None:
        st.error("❌ Please upload and process at least one document first.")
    elif not resolved_api_key:
        st.error("❌ Please provide a valid Groq API key in the sidebar.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Retrieving relevant clauses..."):
                try:
                    result = run_corrective_rag_pipeline(
                        query=user_query,
                        api_key=resolved_api_key,
                        model_name=selected_model,
                    )
                except Exception as exc:
                    st.error(f"❌ An error occurred while processing your query: {exc}")
                    result = None

            if result:
                st.markdown(verdict_badge_html(result["verdict"]), unsafe_allow_html=True)
                st.markdown(result["final_answer"])
                render_sources_expander(result["source_docs"])
                render_verification_expander(result)

                st.session_state.chat_history.append(result)


# --------------------------------------------------------------------------
# 10. FOOTER
# --------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Smart Multi-Document Compliance & Contract Auditor · Built with Streamlit, "
    "LangChain, Groq (Llama 3.3), HuggingFace MiniLM embeddings, and FAISS. "
    "Always have a qualified human reviewer verify compliance-critical outputs."
)
