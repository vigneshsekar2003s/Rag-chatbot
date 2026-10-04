import os
import fitz
import faiss
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer



# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing.")
    st.stop()


# 2. INITIALIZE MODELS

@st.cache_resource
def load_embedding_model():
    """
    Load Sentence Transformer model once.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_groq_client():
    """
    Create Groq API client once.
    """
    return Groq(api_key=GROQ_API_KEY)


embedding_model = load_embedding_model()
groq_client = load_groq_client()


# 3. PDF TEXT EXTRACTION

def extract_text_from_pdf(pdf_file):
    """
    Extract text from uploaded PDF using PyMuPDF.
    """

    pdf_bytes = pdf_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in document:
        page_text = page.get_text()

        if page_text.strip():
            text += page_text + "\n"

    document.close()

    return text


# 4. TEXT CHUNKING

def create_chunks(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    Example:
        chunk 1 = characters 0-500
        chunk 2 = characters 400-900
        chunk 3 = characters 800-1300
    """

    text = text.replace("\n", " ")

    # Remove unnecessary spaces
    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks



# 5. CREATE EMBEDDINGS

def create_embeddings(chunks):
    """
    Convert text chunks into vector embeddings.
    """

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # FAISS works efficiently with float32
    embeddings = embeddings.astype("float32")

    return embeddings


# 6. CREATE FAISS VECTOR DATABASE

def create_faiss_index(embeddings):
    """
    Create FAISS index using cosine similarity.

    Sentence Transformer embeddings are normalized first.
    Therefore inner product = cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Normalize vectors
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    return index



# 7. SEARCH RELEVANT CHUNKS


def search_similar_chunks(query, index, chunks, top_k=4):
    """
    Convert user question into an embedding
    and retrieve the most relevant chunks.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    # Normalize query
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })

    return results



# 8. CREATE RAG PROMPT


def create_prompt(question, retrieved_chunks):
    """
    Build the prompt using retrieved PDF content.
    """

    context = "\n\n".join(
        [
            f"Context {i + 1}:\n{result['text']}"
            for i, result in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say:

"I couldn't find that information in the uploaded document."

Do not make up information.

Keep the answer clear and concise.

Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt



# 9. GENERATE ANSWER USING GROQ


def generate_answer(question, retrieved_chunks):
    """
    Send retrieved context + question to Groq.
    """

    prompt = create_prompt(
        question,
        retrieved_chunks
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise RAG document assistant. "
                    "Only use the provided context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return answer



# 10. STREAMLIT PAGE CONFIGURATION


st.set_page_config(
    page_title="AI Document RAG Chatbot",
    page_icon="assets/rag-icon.png",
    layout="wide"
)



# 11. PAGE TITLE


col1, col2 = st.columns([1, 8])

with col1:
    st.image("assets/rag-icon.png", width=90)

with col2:
    st.title("AI Document RAG Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)

st.divider()



# 12. SIDEBAR


with st.sidebar:

    st.header("📚 Document Settings")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    st.write("")

    st.info(
        """
        **RAG Pipeline**

        PDF
        ↓
        Text Extraction
        ↓
        Chunking
        ↓
        Embeddings
        ↓
        FAISS
        ↓
        Retrieval
        ↓
        Groq
        """
    )



# 13. PROCESS PDF


if uploaded_file:

    # Check whether a new PDF was uploaded
    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file
        != uploaded_file.name
    ):

        with st.spinner("Processing your PDF..."):

            # -----------------------------------------------
            # Extract text
            # -----------------------------------------------

            extracted_text = extract_text_from_pdf(
                uploaded_file
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()

            # -----------------------------------------------
            # Create chunks
            # -----------------------------------------------

            chunks = create_chunks(
                extracted_text,
                chunk_size=500,
                overlap=100
            )

            # -----------------------------------------------
            # Create embeddings
            # -----------------------------------------------

            embeddings = create_embeddings(chunks)

            # -----------------------------------------------
            # Create FAISS index
            # -----------------------------------------------

            index = create_faiss_index(
                embeddings
            )

            # -----------------------------------------------
            # Store everything in session
            # -----------------------------------------------

            st.session_state.processed_file = uploaded_file.name

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.chat_history = []

        st.success(
            f"PDF processed successfully! "
            f"Created {len(chunks)} text chunks."
        )



# 14. DOCUMENT INFORMATION


if "chunks" in st.session_state:

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Text Chunks",
            len(st.session_state.chunks)
        )

    with col2:
        st.metric(
            "Embedding Model",
            "MiniLM"
        )

    st.divider()



# 15. CHAT HISTORY


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )



# 16. CHAT INPUT


question = st.chat_input(
    "Ask a question about your PDF..."
)



# 17. HANDLE USER QUESTION


if question:

    # Make sure PDF has been processed
    if "index" not in st.session_state:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    # -----------------------------------------------
    # Display user question
    # -----------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # Save question
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------

    with st.spinner("Searching document..."):

        retrieved_chunks = search_similar_chunks(
            question,
            st.session_state.index,
            st.session_state.chunks,
            top_k=4
        )

    # -----------------------------------------------
    # Generate answer
    # -----------------------------------------------

    with st.spinner("Generating answer..."):

        answer = generate_answer(
            question,
            retrieved_chunks
        )

    # -----------------------------------------------
    # Display answer
    # -----------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        # Show retrieved context
        with st.expander("🔎 View retrieved context"):

            for i, result in enumerate(
                retrieved_chunks
            ):

                st.markdown(
                    f"**Chunk {i + 1} "
                    f"(similarity: "
                    f"{result['score']:.3f})**"
                )

                st.write(
                    result["text"]
                )

                st.divider()

    # Save answer
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )