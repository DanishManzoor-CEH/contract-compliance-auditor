# Smart Multi-Document Compliance & Contract Auditor

An AI-powered web application for analyzing multiple contracts,
agreements, policies, and compliance documents.

The application combines:

- Streamlit
- LangChain
- Groq LLMs
- HuggingFace Sentence Transformers
- FAISS CPU
- pypdf
- Corrective Retrieval-Augmented Generation
- Citation-aware document analysis

---

## 1. Architecture

The application implements the following multi-stage pipeline:

PDF Upload
    ↓
PDF Text Extraction
    ↓
Page-Level Metadata
    ↓
Recursive Character Chunking
    ↓
HuggingFace Embeddings
    ↓
FAISS Vector Database
    ↓
Similarity Retrieval
    ↓
Groq LLM Draft
    ↓
Corrective Groundedness Verification
    ↓
Citation-Aware Final Answer
    ↓
Source Evidence Viewer


### Processing configuration

Embedding model:

sentence-transformers/all-MiniLM-L6-v2


Chunk size:

1000 characters


Chunk overlap:

150 characters


Retrieval:

Top 4 chunks


Vector database:

FAISS CPU


LLM:

Groq

Default model:

llama-3.3-70b-versatile


Alternative model:

llama3-8b-8192

---

# 2. Features

## Multi-document PDF upload

Upload multiple PDF documents simultaneously.

Examples:

- Contracts
- Vendor agreements
- Privacy policies
- Security policies
- Compliance documents
- Service agreements
- NDAs
- Procurement documents
- Internal policies


## Page-aware metadata

Every extracted page retains:

- File name
- Page number
- Source information


This enables citations such as:

[Vendor_Agreement.pdf | Page 12]


## Recursive chunking

Documents are split using:

RecursiveCharacterTextSplitter

with:

chunk_size = 1000

chunk_overlap = 150


## Semantic retrieval

The application converts document chunks into embeddings using:

sentence-transformers/all-MiniLM-L6-v2

and stores them in a FAISS vector index.

For every question, the application retrieves the four most relevant
chunks.

## Corrective RAG

The system does not simply send retrieved context to the LLM.

It performs two LLM stages:

1. Draft generation
2. Groundedness verification/correction


The verification stage checks whether the draft:

- Uses information from the retrieved evidence
- Contains citations
- Uses valid file/page references
- Contains unsupported claims
- Invents facts
- Makes unsupported legal conclusions

Unsupported statements are removed or rewritten.

---

# 3. Requirements

Recommended Python version:

Python 3.11


The application requires:

- Python
- pip
- Git
- Groq API key

---

# 4. Get a Groq API Key

Create a Groq API key through the Groq developer platform.

Keep the API key private.

Never commit the real API key to GitHub.

---

# 5. Local Installation

## Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/contract-compliance-auditor.git
