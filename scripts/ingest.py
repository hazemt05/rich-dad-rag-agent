import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from bs4 import BeautifulSoup
from src.config import EMBEDDINGS

load_dotenv()

def main():
    print(" Starting Documentation Ingestion Process (LangChain & Qdrant)...")
    
    # Official documentation sources to ingest
    docs_urls = [
        "https://python.langchain.com/docs/concepts/",
        "https://qdrant.tech/documentation/"
    ]
    
    all_chunks = []

    for url in docs_urls:
        print(f"Scraping and loading docs from: {url}")
        try:
            # Use RecursiveUrlLoader to pull web documentation pages
            loader = RecursiveUrlLoader(
                url=url, 
                max_depth=2, 
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
            )
            raw_docs = loader.load()
            
            print(f" Chunking text for {url}...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunked_docs = text_splitter.split_documents(raw_docs)
            all_chunks.extend(chunked_docs)
            print(f" Created {len(chunked_docs)} chunks from {url}.")
        except Exception as e:
            print(f"Error loading {url}: {e}")

    if not all_chunks:
        raise ValueError("No documents were successfully loaded and chunked!")

    print(f"☁️ Uploading total {len(all_chunks)} chunks to Qdrant Cloud in micro-batches...")
    batch_size = 10
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1
        print(f"Uploading micro-batch {current_batch_num} of {total_batches}...")
        
        QdrantVectorStore.from_documents(
            documents=batch,
            embedding=EMBEDDINGS,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="rich_dad_corpus"
        )
        
        if i + batch_size < len(all_chunks):
            time.sleep(15)

    print("\nDocumentation Ingested Successfully into Qdrant Cloud!")

if __name__ == "__main__":
    main()
