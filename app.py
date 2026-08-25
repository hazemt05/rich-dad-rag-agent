import os
import streamlit as st
from dotenv import load_dotenv

# Import your agent functions from src/agents.py
from src.agents import run_researcher, run_reviewer

load_dotenv()

# Streamlit Page Setup
st.set_page_config(page_title="Rich Dad RAG Agents", page_icon="📚", layout="centered")

st.title("📚 Rich Dad, Poor Dad - Multi-Agent Assistant")
st.markdown("Ask any question about the book. Your **Researcher Agent** will query the Qdrant database, and your **Reviewer Agent** will polish the final response!")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history when the app reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from the chat box at the bottom
if user_query := st.chat_input("What would you like to know from the book?"):
    
    # 1. Display user message in chat container
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate Assistant Response using the Multi-Agent Pipeline
    with st.chat_message("assistant"):
        # Create an expandable area to watch the agents work behind the scenes
        with st.status(" Multi-agent workflow in progress...", expanded=True) as status:
            
            # Step 1: Researcher Agent searches Qdrant
            st.write("🔍 **Researcher Agent:** Querying Qdrant vector database...")
            research_draft = run_researcher(user_query)
            
            # Step 2: Reviewer Agent audits and polishes
            st.write(" **Reviewer Agent:** Auditing draft for clarity and accuracy...")
            final_answer = run_reviewer(user_query, research_draft)
            
            status.update(label="Workflow complete!", state="complete", expanded=False)

        # Display the final polished answer in the chat
        st.markdown(final_answer)
        
        # Save assistant response to session history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})