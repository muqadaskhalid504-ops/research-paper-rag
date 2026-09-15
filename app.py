import streamlit as st

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper RAG")

st.write(
    "Upload a research paper and ask questions about its content."
)

st.divider()

st.header("Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success("Research paper uploaded successfully!")
    st.write("File name:", uploaded_file.name)
