import streamlit as st


st.set_page_config(page_title="OmniBrain", page_icon="🧠", layout="wide")

st.title("OmniBrain")
st.caption("Agentic Multi-Modal RAG Orchestrator")

st.subheader("Chat")
st.chat_message("assistant").write("This is a placeholder for the enterprise chat experience.")
user_input = st.chat_input("Ask a multimodal question...")

if user_input:
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write("Retrieval, routing, and citation handling will be added here.")
