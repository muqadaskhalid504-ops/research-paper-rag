 import streamlit as st
import fitz

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

    # Open the uploaded PDF
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # Extract text from all pages
    full_text = ""

    for page in pdf_document:
        text = page.get_text()
        full_text += text + "\n"

    st.subheader("Extracted Text")

    st.text_area(
        "Research Paper Content",
        full_text,
        height=400
    )
