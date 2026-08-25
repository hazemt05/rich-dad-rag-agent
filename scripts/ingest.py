import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from src.config import EMBEDDINGS

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "rich_dad_poor_dad.pdf"

def main():
    print(" Starting Ingestion Process...")
    
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Make sure it's in the data/ folder!")

    print(f"📄 Loading PDF from: {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    raw_documents = loader.load()
    
    print("✂️ Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_docs = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunked_docs)} chunks.")

    print("☁️ Uploading to Qdrant Cloud in ultra-safe micro-batches...")
    
    # Tiny batches of 10 chunks with a 15-second pause to permanently dodge 429 limits
    batch_size = 10
    total_batches = (len(chunked_docs) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1
        print(f"Uploading micro-batch {current_batch_num} of {total_batches}...")
        
        QdrantVectorStore.from_documents(
            documents=batch,
            embedding=EMBEDDINGS,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="rich_dad_corpus"
        )
        
        if i + batch_size < len(chunked_docs):
            print(" Pausing 15 seconds to respect free-tier rate limits...")
            time.sleep(15)

    print(" Ingestion Complete! All data is safely stored in Qdrant Cloud.")

if __name__ == "__main__":
    main()