import os
from typing import List
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma

# =============
# Directories
# =============

DATA_DIR = "rag_data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
os.makedirs(DATA_DIR, exist_ok=True)

# =============
# Model names
# =============

CHAT_MODEL_NAME = "models/gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# =======================
# System prompt
# =======================

SYSTEM_PROMPT = """
You are TaskScapeAI, an AI assistant that helps people answer questions about their schedules, syllabi, and deadlines.

Here are the rules you must follow:

0) If you are not sure a fact comes directly from the document chunks, you MUST say you are unsure. Never guess or invent details.
1) Always answer based on the provided document excerpts only.
2) If the answer is in the document, answer based on it. If not, say you don't know.
3) Do NOT invent dates, deadlines, instructions, events, or assumptions.
4) Some tasks may have higher priority based on percentage of final grade.
5) Whenever possible, quote the exact lines or phrases from the document chunks that support your answer.
6) Keep your answers short, structured, and helpful. Use bullet points when appropriate.
7) Check for exam dates, project deadlines, and important milestones mentioned in the documents.
8) Check exam percentages and grading breakdowns if the user asks about how their final grade is calculated.
9) Make sure to not confuse percentages with dates.


Your goal is to give accurate, document-grounded answers that help the user understand both their academic obligations and their available time..
""".strip()

# Setting up Google Generative AI
def _configure_google():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set in .env")

    genai.configure(api_key=api_key)


# Embedding function
def embed_text(text: str) -> List[float]:
    _configure_google()
    emb = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=text
    )
    return emb["embedding"]

# Gemini to embedding function for LangChain
class Gemini2EmbeddingFunction:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Chroma will call this when ingesting many texts
        return [embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        # Chroma will call this for a single query
        return embed_text(text)


# LLM chat function
def get_llm():
    _configure_google()
    model = genai.GenerativeModel(CHAT_MODEL_NAME)

    def chat(messages: List[dict]):
        prompt = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "System"
            prompt += f"{role}: {msg['content']}\n\n"
        response = model.generate_content(prompt)
        return response.text

    return chat

# Loading and splitting docs
def load_file_to_docs(uploaded_file):
    filename = uploaded_file.name
    path = os.path.join(DATA_DIR, filename)

    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")

    return loader.load()


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# Ingest documents
def ingest_documents(uploaded_files):
    from shutil import rmtree

    all_docs = []

    for file in uploaded_files:
        docs = load_file_to_docs(file)
        if not docs:
            continue

        chunks = split_documents(docs)
        all_docs.extend(chunks)

    if not all_docs:
        if os.path.exists(CHROMA_DIR):
            rmtree(CHROMA_DIR)
        return 0

    if os.path.exists(CHROMA_DIR):
        rmtree(CHROMA_DIR)

    embedding_fn = Gemini2EmbeddingFunction()

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_name="taskscape_documents",
    )

    vectordb.persist()
    return len(all_docs)

# Load vector store
def _load_vector_store():
    if not os.path.exists(CHROMA_DIR):
        return None
    return Chroma(
        embedding_function=Gemini2EmbeddingFunction(),
        persist_directory=CHROMA_DIR,
        collection_name="taskscape_documents",
    )

# Answer question
def answer_question(query: str) -> str:
    vectordb = _load_vector_store()
    if vectordb is None:
        return "No documents ingested yet."

    docs = vectordb.similarity_search(query, k=5)
    if not docs:
        return "I couldn't find anything relevant in your uploaded documents."

    context = "\n\n".join(f"[DOC {i+1}]\n{d.page_content}" for i, d in enumerate(docs))

    prompt = f"""
{SYSTEM_PROMPT}

User question:
{query}

Relevant excerpts:
{context}

Answer using ONLY the excerpts above.
"""

    llm = get_llm()
    result = llm([{"role": "user", "content": prompt}])
    return result
