import os
import streamlit as st
from dotenv import load_dotenv
from src.agents import run_researcher, run_reviewer

load_dotenv()

st.set_page_config(page_title="RAG Multi-Agent Assistant", page_icon="", layout="centered")

st.title("LangChain & Qdrant Documentation Assistant")
st.markdown("Query the official documentation. Watched over by a **Researcher Agent** and audited by a strict **Reviewer Agent**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Retrieved Source Passages"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}:**\n```text\n{doc.page_content}\n```")

if user_query := st.chat_input("Ask a technical question about LangChain or Qdrant..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("Multi-agent pipeline running...", expanded=True) as status:
            st.write(" **Researcher Agent:** Retrieving document chunks...")
            research_result = run_researcher(user_query)
            draft = research_result["draft"]
            sources = research_result["source_docs"]
            
            st.write("**Reviewer Agent:** Auditing claims and checking source backing...")
            final_reviewed_output = run_reviewer(user_query, draft, sources)
            
            status.update(label="Workflow and Audit complete!", state="complete", expanded=False)

        st.markdown(final_reviewed_output)
        
        if sources:
            with st.expander(" View Retrieved Source Passages"):
                for idx, doc in enumerate(sources):
                    st.markdown(f"**Source {idx+1}:**\n```text\n{doc.page_content}\n```")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_reviewed_output,
            "sources": sources
        })
