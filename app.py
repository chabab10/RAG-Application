import streamlit as st
from rag.loader import load_pdfs
from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore
from rag.rag_chain import answer_question

st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG – Chat with your PDFs")

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🔄 Index PDFs"):
        with st.spinner("Indexing PDFs..."):
            docs = load_pdfs(uploaded_files)
            chunks = split_documents(docs)
            st.session_state.vectordb = create_vectorstore(chunks)
        st.success("✅ PDFs indexed")

if st.session_state.vectordb:
    question = st.text_input("Ask a question")

    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(
                st.session_state.vectordb,
                question
            )
        st.subheader("Answer")
        st.write(answer)
