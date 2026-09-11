"""
Smart Multi-Document Compliance & Contract Auditor

Architecture:
    PDF Upload
        ↓
    PDF Text Extraction
        ↓
    Metadata Preservation
        ↓
    Recursive Chunking
        ↓
    HuggingFace Embeddings
        ↓
    FAISS Vector Store
        ↓
    Similarity Retrieval
        ↓
    Groq LLM Draft Answer
        ↓
    Corrective Groundedness Verification
        ↓
    Citation-Aware Final Answer
        ↓
    Source Evidence UI

Technology:
    - Streamlit
    - LangChain
    - Groq
    - HuggingFace Sentence Transformers
    - FAISS CPU
    - pypdf
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------

load_dotenv()

APP_TITLE = "Smart Multi-Document Compliance & Contract Auditor"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_RETRIEVAL_K = 4

SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
]


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            padding-top: 1.5rem;
        }

        .app-header {
            padding: 1.5rem 1.75rem;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                rgba(31, 41, 55, 0.98),
                rgba(17, 24, 39, 0.98)
            );
            color: white;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .app-header h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
        }

        .app-header p {
            margin-top: 0.6rem;
            margin-bottom: 0;
            color: #d1d5db;
            font-size: 1rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            background-color: rgba(128,128,128,0.05);
            text-align: center;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .metric-label {
            font-size: 0.82rem;
            opacity: 0.75;
        }

        .source-card {
            padding: 0.9rem;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 0.75rem;
            background-color: rgba(128,128,128,0.04);
        }

        .citation {
            font-weight: 600;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(245, 158, 11, 0.45);
            background-color: rgba(245, 158, 11, 0.08);
        }

        .success-box {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(34, 197, 94, 0.45);
            background-color: rgba(34, 197, 94, 0.08);
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="app-header">
        <h1>📑 {APP_TITLE}</h1>
        <p>
            Analyze multiple contracts, policies, compliance documents and
            agreements using corrective retrieval-augmented generation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "document_stats" not in st.session_state:
    st.session_state.document_stats = {
        "files": 0,
        "pages": 0,
        "chunks": 0,
    }

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_secret_or_env(name: str) -> str:
    """
    Retrieve a configuration value in the following order:

    1. Streamlit secrets
    2. Environment variable

    Returns an empty string when the value is unavailable.
    """

    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, "").strip()


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the HuggingFace embedding model.

    Streamlit cache_resource prevents the embedding model from being
    downloaded and initialized for every interaction.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text without destroying meaningful content.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_documents(uploaded_files: list[Any]) -> list[Document]:
    """
    Extract text from every uploaded PDF.

    Each PDF page becomes a LangChain Document so that page-level
    metadata can be preserved.

    Metadata:
        source
        file_name
        page
        page_number
    """

    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(
                f"Unable to read '{uploaded_file.name}' as a PDF: {exc}"
            ) from exc

        for page_index, page in enumerate(reader.pages):
            try:
                extracted_text = page.extract_text() or ""
            except Exception:
                extracted_text = ""

            extracted_text = clean_text(extracted_text)

            if not extracted_text:
                continue

            page_number = page_index + 1

            documents.append(
                Document(
                    page_content=extracted_text,
                    metadata={
                        "source": uploaded_file.name,
                        "file_name": uploaded_file.name,
                        "page": page_number,
                        "page_number": page_number,
                    },
                )
            )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split page-level documents into overlapping chunks.

    Required configuration:
        chunk_size=1000
        chunk_overlap=150
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index

    return chunks


def build_vector_store(chunks: list[Document]) -> FAISS:
    """
    Create a FAISS vector index from document chunks.
    """

    embeddings = load_embedding_model()

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def get_llm(api_key: str, model_name: str) -> ChatGroq:
    """
    Create a Groq chat model.

    Temperature is kept at zero because compliance and contract analysis
    should prioritize deterministic, evidence-based responses.
    """

    if not api_key:
        raise ValueError(
            "Groq API key is required. Add GROQ_API_KEY to Streamlit "
            "Secrets, .env, or enter it in the sidebar."
        )

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0,
        max_retries=2,
    )


def format_retrieved_context(documents: list[Document]) -> str:
    """
    Convert retrieved documents into a citation-aware context block.

    Every chunk receives a stable evidence identifier.
    """

    context_parts: list[str] = []

    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get(
            "file_name",
            document.metadata.get("source", "Unknown"),
        )

        page_number = document.metadata.get(
            "page_number",
            document.metadata.get("page", "Unknown"),
        )

        context_parts.append(
            f"""
[EVIDENCE {index}]
File Name: {file_name}
Page Number: {page_number}
Citation: [{file_name} | Page {page_number}]

Text:
{document.page_content}
[/EVIDENCE {index}]
""".strip()
        )

    return "\n\n".join(context_parts)


def generate_draft_answer(
    llm: ChatGroq,
    user_query: str,
    retrieved_documents: list[Document],
) -> str:
    """
    First-generation RAG step.

    The LLM is explicitly instructed to answer only from retrieved
    evidence and provide file/page citations.
    """

    context = format_retrieved_context(retrieved_documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a careful contract and compliance analysis assistant.

Your job is to answer the user's question using ONLY the supplied
document evidence.

Rules:

1. Do not use outside knowledge as factual evidence.
2. Do not invent clauses, dates, obligations, parties, penalties,
   percentages, deadlines, or legal conclusions.
3. If the evidence does not contain enough information, explicitly say
   that the supplied documents do not provide sufficient evidence.
4. Every material factual statement must include a citation.
5. Citations MUST use this format:

   [File Name | Page N]

6. When multiple documents support a statement, cite each relevant source.
7. Distinguish between:
   - What the document explicitly says.
   - What can reasonably be inferred.
   - What cannot be determined from the supplied documents.
8. For compliance findings, identify the relevant requirement, evidence,
   risk/issue, and recommendation when the evidence supports them.
9. Never claim that a contract is legally valid, invalid, enforceable,
   compliant, or non-compliant unless the supplied evidence explicitly
   supports such a conclusion.
10. Keep the answer professional and concise.

Retrieved evidence:
{context}
""",
            ),
            (
                "human",
                "User question:\n{question}",
            ),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": user_query,
        }
    )

    return response.content.strip()


def verify_and_correct_answer(
    llm: ChatGroq,
    user_query: str,
    draft_answer: str,
    retrieved_documents: list[Document],
) -> str:
    """
    Corrective verification stage.

    The second LLM call acts as a groundedness checker/editor.

    It checks:
        - whether claims are supported
        - whether citations exist
        - whether citations point to retrieved evidence
        - whether unsupported claims should be removed
        - whether missing evidence should be explicitly disclosed

    The verifier returns the corrected final answer rather than merely
    returning a pass/fail score.
    """

    context = format_retrieved_context(retrieved_documents)

    verification_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the final groundedness and citation verifier for a
document-analysis system.

You will receive:
1. The user's question.
2. A draft answer.
3. The exact evidence retrieved from the uploaded documents.

Your task is to produce the FINAL corrected answer.

Verification rules:

A. Compare every factual/material claim in the draft against the
   retrieved evidence.

B. Remove or rewrite claims that cannot be directly supported.

C. Never introduce information from your general knowledge.

D. Every material factual statement must contain a citation in exactly
   this form:

   [File Name | Page N]

E. A citation is valid only if that exact file and page occur in the
   supplied evidence.

F. If evidence is insufficient, clearly state:
   "The supplied documents do not provide sufficient evidence to
   determine this."

G. Preserve useful supported findings.

H. Do not fabricate citations.

I. Do not claim legal enforceability or legal validity unless the
   supplied evidence itself establishes that fact.

J. Return ONLY the corrected final answer in Markdown.
Do not discuss this verification process.

Retrieved evidence:
{context}

Draft answer:
{draft_answer}
""",
            ),
            (
                "human",
                "Original user question:\n{question}",
            ),
        ]
    )

    chain = verification_prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "draft_answer": draft_answer,
            "question": user_query,
        }
    )

    corrected_answer = response.content.strip()

    if not corrected_answer:
        return draft_answer

    return corrected_answer


def retrieve_documents(
    vector_store: FAISS,
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks from FAISS.
    """

    return vector_store.similarity_search(
        query,
        k=k,
    )


def render_source_evidence(documents: list[Document]) -> None:
    """
    Render retrieved evidence in an expandable UI section.
    """

    with st.expander(
        f"🔎 Retrieved Source Evidence ({len(documents)} chunks)",
        expanded=False,
    ):
        st.caption(
            "These are the exact text chunks supplied to the LLM during "
            "answer generation and verification."
        )

        for index, document in enumerate(documents, start=1):
            file_name = document.metadata.get(
                "file_name",
                document.metadata.get("source", "Unknown"),
            )

            page_number = document.metadata.get(
                "page_number",
                document.metadata.get("page", "Unknown"),
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                index,
            )

            st.markdown(
                f"""
                <div class="source-card">
                    <strong>Evidence {index}</strong><br>
                    <span class="citation">
                        📄 {file_name} | Page {page_number}
                    </span><br>
                    Chunk ID: {chunk_id}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.code(
                document.page_content,
                language="text",
            )


def render_document_stats() -> None:
    """
    Render processing statistics.
    """

    stats = st.session_state.document_stats

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats["files"]}</div>
                <div class="metric-label">PDF Files</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats["pages"]}</div>
                <div class="metric-label">Pages Extracted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats["chunks"]}</div>
                <div class="metric-label">Indexed Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Auditor Configuration")

    st.markdown(
        """
        Upload one or more PDF documents. The application creates a
        temporary FAISS index in the current Streamlit session.
        """
    )

    uploaded_files = st.file_uploader(
        "📄 Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload contracts, policies, agreements, compliance documents, or related PDFs.",
    )

    st.divider()

    configured_api_key = get_secret_or_env("GROQ_API_KEY")

    api_key = st.text_input(
        "🔑 Groq API Key",
        value="",
        type="password",
        placeholder=(
            "Using configured secret"
            if configured_api_key
            else "Enter your Groq API key"
        ),
        help=(
            "The key can come from Streamlit Secrets, .env, or this field."
        ),
    )

    effective_api_key = api_key.strip() or configured_api_key

    model_name = st.selectbox(
        "🤖 Groq Model",
        options=SUPPORTED_MODELS,
        index=0,
        help="Select the Groq model used for generation and verification.",
    )

    st.divider()

    process_documents = st.button(
        "🚀 Process Documents",
        type="primary",
        use_container_width=True,
    )

    if st.session_state.vector_store is not None:
        st.success("Vector index ready")

    st.caption(
        "Embeddings: all-MiniLM-L6-v2\n\n"
        "Vector DB: FAISS CPU\n\n"
        "Retrieval: Top 4 chunks\n\n"
        "Chunk size: 1000\n\n"
        "Chunk overlap: 150"
    )


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

if process_documents:
    if not uploaded_files:
        st.warning("Please upload at least one PDF document.")
    else:
        with st.status(
            "Processing documents...",
            expanded=True,
        ) as status:
            try:
                st.write("📖 Extracting PDF text...")

                page_documents = extract_pdf_documents(
                    uploaded_files
                )

                if not page_documents:
                    raise ValueError(
                        "No extractable text was found in the uploaded PDFs. "
                        "The files may contain scanned images instead of "
                        "machine-readable text."
                    )

                st.write(
                    f"✅ Extracted text from {len(page_documents)} pages."
                )

                st.write("✂️ Splitting documents into overlapping chunks...")

                chunks = split_documents(page_documents)

                if not chunks:
                    raise ValueError(
                        "Document splitting produced no usable chunks."
                    )

                st.write(
                    f"✅ Created {len(chunks)} chunks."
                )

                st.write(
                    "🧠 Loading HuggingFace embedding model..."
                )

                # Explicitly initialize here so failures happen during
                # processing rather than during a later query.
                load_embedding_model()

                st.write("🔎 Building FAISS vector index...")

                vector_store = build_vector_store(chunks)

                # Store everything required by the current session.
                st.session_state.vector_store = vector_store
                st.session_state.processed_documents = [
                    file.name for file in uploaded_files
                ]
                st.session_state.document_chunks = chunks
                st.session_state.document_stats = {
                    "files": len(uploaded_files),
                    "pages": len(page_documents),
                    "chunks": len(chunks),
                }

                st.session_state.last_query = ""
                st.session_state.last_sources = []

                status.update(
                    label="Documents processed successfully!",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                status.update(
                    label="Document processing failed",
                    state="error",
                    expanded=True,
                )

                st.error(
                    f"Processing error: {exc}"
                )


# ---------------------------------------------------------------------------
# Main document status
# ---------------------------------------------------------------------------

if st.session_state.vector_store is not None:
    st.subheader("📊 Indexed Document Collection")

    if st.session_state.processed_documents:
        st.write(
            " **Processed files:** "
            + ", ".join(st.session_state.processed_documents)
        )

    render_document_stats()

    st.divider()


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

st.subheader("🔍 Ask Your Compliance or Contract Question")

st.markdown(
    """
    Ask questions such as:

    - What termination obligations are defined in the agreement?
    - Which party is responsible for data protection?
    - Identify clauses that may create compliance risks.
    - What are the payment deadlines?
    - Compare the confidentiality requirements across the uploaded contracts.
    - Does the policy mention an incident notification deadline?
    """
)

query = st.text_area(
    "Your question",
    placeholder=(
        "Example: Identify the termination obligations in the uploaded "
        "contracts and cite the relevant file and page."
    ),
    height=120,
)

analyze_button = st.button(
    "🔎 Analyze Documents",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# RAG + Corrective Verification
# ---------------------------------------------------------------------------

if analyze_button:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    if st.session_state.vector_store is None:
        st.warning(
            "Please upload and process documents before asking a question."
        )
        st.stop()

    if not effective_api_key:
        st.error(
            "Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets "
            "or .env, or enter the key in the sidebar."
        )
        st.stop()

    try:
        with st.status(
            "Running corrective RAG analysis...",
            expanded=True,
        ) as status:

            # ---------------------------------------------------------------
            # Stage 1: Retrieval
            # ---------------------------------------------------------------

            st.write(
                "🔎 Stage 1/3 — Retrieving the most relevant evidence..."
            )

            retrieved_documents = retrieve_documents(
                st.session_state.vector_store,
                query.strip(),
                k=DEFAULT_RETRIEVAL_K,
            )

            if not retrieved_documents:
                status.update(
                    label="No relevant evidence found",
                    state="complete",
                    expanded=False,
                )

                st.warning(
                    "No relevant document chunks were retrieved for this query."
                )

                st.stop()

            # ---------------------------------------------------------------
            # Stage 2: Draft generation
            # ---------------------------------------------------------------

            st.write(
                "🤖 Stage 2/3 — Generating an evidence-grounded draft..."
            )

            llm = get_llm(
                api_key=effective_api_key,
                model_name=model_name,
            )

            draft_answer = generate_draft_answer(
                llm=llm,
                user_query=query.strip(),
                retrieved_documents=retrieved_documents,
            )

            # ---------------------------------------------------------------
            # Stage 3: Corrective verification
            # ---------------------------------------------------------------

            st.write(
                "🛡️ Stage 3/3 — Verifying claims and correcting citations..."
            )

            final_answer = verify_and_correct_answer(
                llm=llm,
                user_query=query.strip(),
                draft_answer=draft_answer,
                retrieved_documents=retrieved_documents,
            )

            st.session_state.last_query = query.strip()
            st.session_state.last_sources = retrieved_documents

            status.update(
                label="Analysis and verification completed",
                state="complete",
                expanded=False,
            )

        # ---------------------------------------------------------------
        # Final output
        # ---------------------------------------------------------------

        st.subheader("🧠 Verified Analysis")

        st.markdown(final_answer)

        st.markdown(
            """
            <div class="success-box">
                <strong>Groundedness verification completed.</strong><br>
                The final response was checked against the retrieved
                document evidence before being displayed.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        render_source_evidence(retrieved_documents)

    except Exception as exc:
        st.error(
            f"Unable to complete the analysis: {exc}"
        )

        with st.expander("Technical details"):
            st.exception(exc)


# ---------------------------------------------------------------------------
# Initial state information
# ---------------------------------------------------------------------------

if st.session_state.vector_store is None:
    st.info(
        "👈 Upload one or more PDF documents in the sidebar and click "
        "**Process Documents** to build the searchable knowledge base."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "Smart Multi-Document Compliance & Contract Auditor • "
    "Streamlit + LangChain + Groq + HuggingFace + FAISS"
)

st.caption(
    "⚠️ This tool provides document-grounded analysis and should not be "
    "treated as legal advice."
)
