import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import EMBEDDINGS

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# AGENT 1: THE RESEARCHER (Returns Answer + Source Docs)
# ==========================================
def run_researcher(query: str):
    print(f"[Researcher] Searching Qdrant database for: '{query}'...")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=EMBEDDINGS,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="rich_dad_corpus"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Retrieve docs explicitly so we can pass them to the UI
    retrieved_docs = retriever.invoke(query)
    context_text = format_docs(retrieved_docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        "You are an expert research assistant. Use the following pieces "
        "of retrieved context from the official documentation to answer the question. "
        "If you don't know the answer or if it's not in the context, say that "
        "you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": query})

    return {
        "draft": answer,
        "source_docs": retrieved_docs
    }


# ==========================================
# AGENT 2: THE REVIEWER (Strict Fact-Checker & Verdict)
# ==========================================
def run_reviewer(original_query: str, research_draft: str, source_docs: list):
    print(f" [Reviewer] Auditing draft against source passages...")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    context_str = format_docs(source_docs)

    review_prompt = ChatPromptTemplate.from_template(
        "You are a strict editorial reviewer and technical fact-checker. "
        "Your job is to audit the Researcher's draft against the provided Source Passages.\n\n"
        "Original Question: {query}\n\n"
        "Source Passages:\n{context}\n\n"
        "Research Draft:\n{draft}\n\n"
        "Instructions:\n"
        "1. Verify whether every claim in the Research Draft is directly backed by the Source Passages.\n"
        "2. If there are unsupported claims or hallucinations, explicitly REJECT or FLAG them and correct them.\n"
        "3. Provide a clear verdict (e.g., 'VERDICT: APPROVED' or 'VERDICT: REJECTED/CORRECTED') followed by your polished response."
    )

    review_chain = review_prompt | llm | StrOutputParser()
    final_output = review_chain.invoke({
        "query": original_query, 
        "context": context_str, 
        "draft": research_draft
    })
    
    return final_output
