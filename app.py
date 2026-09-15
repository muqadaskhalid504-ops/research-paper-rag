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


def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


if uploaded_file is not None:

    st.success("Research paper uploaded successfully!")

    # Read PDF
    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    # Extract text
    full_text = ""

    for page in pdf_document:
        text = page.get_text()
        full_text += text + "\n"

    # Create chunks
    chunks = create_chunks(full_text)

    st.subheader("Paper Information")

    st.write("Total characters:", len(full_text))
    st.write("Total chunks:", len(chunks))

    st.subheader("First Chunk")

    st.text_area(
        "Chunk 1",
        chunks[0] if chunks else "No text found.",
        height=300
    )
