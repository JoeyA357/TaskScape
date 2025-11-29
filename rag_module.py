# rag_module.py
#
# RAG without embeddings (because your AI Studio key has 0 embedding quota).
# Uses Gemini 2.0 Flash to:
#   - select relevant document chunks
#   - answer questions about schedules, syllabi, and project docs
# Includes a simple "intent detector" to switch between:
#   - normal document Q&A
#   - time-allocation / free-slot suggestions

import json
import os
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import SystemMessage, HumanMessage


# =========================
# Config & Globals
# =========================

DATA_DIR = "rag_data"
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "gemini-2.0-flash"


SYSTEM_PROMPT = """
You are TaskScapeAI, an AI assistant that helps people answer questions about their schedules, syllabi, and deadlines.

Here are the rules you must follow:

0) If you are not sure a fact comes directly from the document chunks, you MUST say you are unsure. Never guess or invent details.
1) Use only the provided document excerpts to answer the question.
2) If the documents do not contain the answer, say: "I couldn't find anything relevant in your uploaded documents."
3) Do NOT invent dates, deadlines, instructions, events, or assumptions.
4) You should be aware of the user's free time based on the documents provided. This will help plan appropriately when they ask about scheduling.
5) Urgency is important. If asked, the closest deadlines should be prioritized.
6) Some tasks may have higher priority based on importance — not just deadlines. If the user specifies a task is higher importance, prioritize that task.
7) Whenever possible, quote the exact lines or phrases from the document chunks that support your answer.
8) Keep your answers short, structured, and helpful. Use bullet points when appropriate.

Your goal is to give accurate, document-grounded answers that help the user understand both their academic obligations and their available time.
"""


SCHEDULE_HINT = """
The documents appear to be a "Student Schedule by Day and Time" page.

Columns correspond to days of the week (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday).
Rows correspond to time slots (8am, 9am, 10am, 11am, 12pm, 1pm, 2pm, 3pm, 4pm, etc.).

Each block such as:

"COE 414-32
15870 Class
9:00 am-9:50 am
204 0403"

means:
- course code and section (COE 414-32),
- then the time range (9:00 am-9:50 am),
- then the room number (204 0403).

When answering questions, focus on:
- which courses happen on which days,
- and the time ranges and free time around them.
"""


def get_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model client using the GOOGLE_API_KEY from .env."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file at the project root."
        )

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )


# =========================
# Helpers for docs & chunks
# =========================

def load_file_to_text(uploaded_file) -> str:
    """
    Save the uploaded file into rag_data/ and extract its text.
    Supports PDF and TXT for now.
    """
    filename = uploaded_file.name
    path = os.path.join(DATA_DIR, filename)

    # Save uploaded file bytes
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    # Choose loader based on extension
    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        # fallback: treat as plain text
        loader = TextLoader(path)

    docs = loader.load()
    raw_text = " \n".join(d.page_content for d in docs)

    # Basic cleanup: collapse all whitespace to single spaces
    cleaned = " ".join(raw_text.split())
    return cleaned


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks for RAG.

    For short docs like schedules, we keep it as a single chunk.
    """
    if len(text) < 2000:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_doc_hint(chunks: List[str]) -> str:
    """
    Look at the document text and return an extra hint
    (e.g., schedule-specific hint) if appropriate.
    """
    all_text = " ".join(chunks).lower()

    if "student schedule by day and time" in all_text:
        return SCHEDULE_HINT

    # You can extend this later for syllabi / project docs if you want.
    return ""


# =========================
# Public API for Streamlit
# =========================

def ingest_documents(uploaded_files: List) -> int:
    """
    Ingest uploaded files:
      - extract text
      - chunk it
      - store chunks into rag_data/chunks.json

    Returns:
        number of chunks created.
    """
    all_chunks: List[str] = []

    for file in uploaded_files:
        text = load_file_to_text(file)
        if not text.strip():
            # No usable text extracted (e.g., scanned PDF with no OCR)
            print(f"[RAG] Warning: no text extracted from {file.name}")
            continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    if not all_chunks:
        # Nothing to store
        chunks_path = os.path.join(DATA_DIR, "chunks.json")
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        return 0

    # Persist chunks locally so we can use them later in answer_question
    with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    return len(all_chunks)


def answer_question(query: str) -> str:
    """
    Main RAG function used by the UI.

    - Loads chunks from rag_data/chunks.json
    - Uses Gemini to select relevant chunks
    - Then uses Gemini again to answer, with two modes:

      * Normal document Q&A
      * Time allocation (finding free slots in schedule) when the question
        clearly asks about "when can I work / where can I allocate time".
    """
    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        return "No documents ingested yet."

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[str] = json.load(f)

    if not chunks:
        return "No text was extracted from your documents."

    # Decide whether the user is asking about time allocation vs normal Q&A
    q_lower = query.lower()
    time_keywords = [
        "free time",
        "when can i work",
        "where can i allocate time",
        "fit this in my schedule",
        "when do i have time",
        "schedule this",
        "time for this project",
        "time for this assignment",
        "time to work on this",
    ]
    is_time_allocation = any(kw in q_lower for kw in time_keywords)

    # Build optional extra hint (e.g., schedule-specific)
    extra_hint = build_doc_hint(chunks)

    # Build a numbered string for chunks so Gemini can read them more easily
    numbered_chunks = []
    for i, ch in enumerate(chunks, start=1):
        numbered_chunks.append(f"[CHUNK {i}]\n{ch}\n")
    chunks_text_for_llm = "\n".join(numbered_chunks)

    # === Step 1: select relevant chunks ===
    selection_prompt = f"""
{extra_hint}

User question:
{query}

Here are numbered document chunks:

{chunks_text_for_llm}

Your first task is to pick ONLY the chunks that are relevant for answering the question.
Return those relevant chunks as plain text. If none are relevant, return an empty list or an explanation that nothing is relevant.
"""

    llm = get_llm()

    selection_response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=selection_prompt),
        ]
    )

    # Make sure whatever the model outputs becomes a clean string
    selected_chunks_text = str(selection_response.content)

    # === Step 2: answer based on the selected chunks ===
    if is_time_allocation:
        # MODE B — time allocation / free-slot suggestions
        answer_prompt = f"""
User question:
{query}

Relevant document passages (schedule blocks):
{selected_chunks_text}

Your task:
- Read the schedule info carefully.
- Identify concrete free time slots where no classes or events are scheduled.
- Suggest blocks where the user can work on their project or task.

Rules:
- Answer ONLY with time suggestions and very short notes (no long explanations).
- Do NOT describe what the project is about.
- Do NOT talk about tools, LLMs, or implementation details.
- Use a clear bullet list format like:
  - Monday 10:00–12:00 — good for deep work
  - Wednesday 15:00–17:00 — can work on project report
  - Friday 09:00–11:00 — free slot

If you truly cannot infer any free time from the schedule, say:
"I couldn't identify any clear free time blocks from your documents."
"""
    else:
        # MODE A — normal document Q&A (projects, syllabi, deadlines, etc.)
        answer_prompt = f"""
User question:
{query}

Relevant document passages:
{selected_chunks_text}

Using ONLY the information above:
- Answer the question clearly and concisely.
- If the question is about project requirements, explain the key tasks, components, or deadlines mentioned in the documents.
- If the question is about exams or deadlines, list the relevant dates and what they correspond to.
- If the documents do not contain the answer, say:
  "I couldn't find anything relevant in your uploaded documents."
"""

    final_response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=answer_prompt),
        ]
    )

    return final_response.content
