import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import EMBEDDINGS

# Suppress minor SDK warnings in the terminal
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

def format_docs(docs):
    """Helper function to combine retrieved chunks into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# AGENT 1: THE RESEARCHER
# ==========================================
def run_researcher(query: str):
    print(f" [Researcher] Searching Qdrant database for: '{query}'...")

    # 1. Connect to Qdrant Cloud
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=EMBEDDINGS,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="rich_dad_corpus"
    )

    # 2. Setup the retriever to grab top 3 chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Setup Gemini Brain
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # 4. Prompt rules for finding info
    prompt = ChatPromptTemplate.from_template(
        "You are an expert research assistant. Use the following pieces "
        "of retrieved context from the book to answer the question. "
        "If you don't know the answer or if it's not in the context, say that "
        "you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    # 5. Build pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Run and return draft
    return rag_chain.invoke(query)


# ==========================================
# AGENT 2: THE REVIEWER
# ==========================================
def run_reviewer(original_query: str, research_draft: str):
    print(f" [Reviewer] Auditing and polishing the research draft...")

    # 1. Setup Gemini Brain for the Reviewer
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # 2. Prompt rules for editorial quality control
    review_prompt = ChatPromptTemplate.from_template(
        "You are a strict editorial reviewer and fact-checker. "
        "Review the following research draft that was generated to answer a user's question.\n\n"
        "Original Question: {query}\n\n"
        "Research Draft:\n{draft}\n\n"
        "Your task: Check the draft for clarity, conciseness, and accuracy based on the context. "
        "Refine and polish the output into a professional, well-structured final response. "
        "Do not invent new facts."
    )

    # 3. Build simple chain
    review_chain = review_prompt | llm | StrOutputParser()

    # 4. Run and return final polished output
    return review_chain.invoke({"query": original_query, "draft": research_draft})


# ==========================================
# ORCHESTRATION (Running both together)
# ==========================================
if __name__ == "__main__":
    test_question = "how to build a house?"
    
    # Step 1: Agent 1 researches and drafts
    research_result = run_researcher(test_question)
    print("\n [Researcher Draft]:")
    print(research_result)
    
    print("\n" + "="*50 + "\n")
    
    # Step 2: Agent 2 reviews and polishes
    final_output = run_reviewer(test_question, research_result)
    print("\n [Reviewer Final Output]:")
    print(final_output)
