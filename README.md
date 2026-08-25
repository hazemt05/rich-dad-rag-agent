#  Rich Dad, Poor Dad - Multi-Agent RAG Assistant

An intelligent, multi-agent Retrieval-Augmented Generation (RAG) web application built using Python, LangChain, Qdrant Cloud, Google Gemini (`gemini-3.6-flash`), and Streamlit.

##  Architectural Blueprint

This project implements a **two-agent collaborative workflow** to ensure high-accuracy responses and eliminate hallucinations when querying text from *Rich Dad, Poor Dad*:

1. **Ingestion Pipeline (`src/ingest.py`):** Loads the PDF book, splits text dynamically using `RecursiveCharacterTextSplitter` with intelligent chunking and overlaps, converts text into dense vector embeddings via Google GenAI, and streams data into **Qdrant Cloud** in micro-batches (with rate-limit mitigation).
2. **The Researcher Agent (`src/agents.py`):** Acts as the retrieval engine. It queries the Qdrant vector database, extracts the top $k$ relevant context chunks, and drafts a factual answer strictly grounded in the book's text.
3. **The Reviewer Agent (`src/agents.py`):** Acts as an editorial supervisor. It audits the Researcher's draft for clarity, structure, and conciseness, refining it into a polished final response without inventing new facts.
4. **Interactive UI (`app.py`):** A clean Streamlit chat application that visualizes the multi-agent workflow in real time.

---

##  Project Structure

```text
AI-AGENT-PROJECT/
├── data/                  # Place your source PDFs here (e.g., rich_dad_poor_dad.pdf)
├── src/
│   ├── config.py          # Centralized configuration and embeddings setup
│   ├── ingest.py          # PDF document loader, text splitter, and Qdrant uploader
│   └── agents.py          # Researcher and Reviewer agent logic (LCEL pipelines)
├── .env                   # Local API keys (Ignored by Git)
├── .env.example           # Template for environment variables
├── app.py                 # Streamlit web interface
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation