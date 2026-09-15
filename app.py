import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

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


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


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

    # Load embedding model
    with st.spinner("Creating embeddings..."):

        model = load_embedding_model()

        embeddings = model.encode(
            chunks,
            show_progress_bar=False
        )

    embeddings = np.array(embeddings).astype("float32")

    st.success("Embeddings created successfully!")

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success("FAISS vector database created successfully!")

    st.divider()

    # Question section
    st.header("💬 Ask a Question")

    question = st.text_input(
        "Enter your question about the research paper:"
    )

    if question:

        # Convert question into embedding
        question_embedding = model.encode(
            [question]
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")

        # Search FAISS
        number_of_results = 3

        distances, indices = index.search(
            question_embedding,
            number_of_results
        )

        # Get relevant chunks
        relevant_chunks = []

        for index_number in indices[0]:

            relevant_chunks.append(
                chunks[index_number]
            )

        # Combine chunks
        context = "\n\n".join(
            relevant_chunks
        )

        # Groq client
        client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

        # Prompt for the AI
        prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the
information provided in the research paper context.

If the answer is not available in the context,
say:

"This information is not available in the
uploaded research paper."

Do not make up information.

Research Paper Context:
{context}

User Question:
{question}
"""

        with st.spinner("Generating answer..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2
            )

        answer = response.choices[0].message.content

        st.subheader("🤖 Answer")

        st.write(answer)

        st.subheader("📚 Sources")

        for i, index_number in enumerate(indices[0]):

            st.markdown(
                f"### Source {i + 1}"
            )

            st.write(
                chunks[index_number]
            )
