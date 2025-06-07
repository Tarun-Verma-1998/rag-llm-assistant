# app.py

import streamlit as st

st.set_page_config(page_title="RAG + LLM Judge", layout="centered")

st.title(" RAG + LLM-as-a-Judge Assistant")
st.markdown("Ask a question and get a real-time answer using retrieval, reranking, and generation!")

# Input box
query = st.text_input("Enter your question:")

# Submit
if st.button("Get Answer") and query:
    st.write("🔍 Processing your query...")
    # We will plug in backend logic in the next step
