import os
import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from openai import OpenAI


# -------------------------
# Vector Database
# -------------------------

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function
)


# -------------------------
# OpenAI Client
# -------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# -------------------------
# Extract PDF Text
# -------------------------

def extract_text_from_pdf(file_path: str) -> str:

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -------------------------
# Split Text into Chunks
# -------------------------

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -------------------------
# Add PDF to Database
# -------------------------

def add_pdf(file_path: str):

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise ValueError(
            "Could not extract text from this PDF."
        )

    chunks = chunk_text(text)

    ids = []

    metadatas = []

    for index, chunk in enumerate(chunks):

        ids.append(
            str(uuid.uuid4())
        )

        metadatas.append(
            {
                "source": os.path.basename(file_path),
                "chunk": index
            }
        )

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)


# -------------------------
# Retrieve Relevant Context
# -------------------------

def retrieve_context(
    question: str,
    n_results: int = 5
):

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    return documents


# -------------------------
# Ask the AI
# -------------------------

def ask_question(question: str):

    documents = retrieve_context(question)

    if not documents:

        return "No relevant information was found."

    context = "\n\n---\n\n".join(
        documents
    )

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY
the context provided below.

If the answer is not contained in
the context, say:

"I could not find the answer in the uploaded documents."

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
