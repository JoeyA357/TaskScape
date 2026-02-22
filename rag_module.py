import os
import re
import json
from datetime import datetime, time as dt_time
from typing import List, Dict, Optional, Any
from collections import defaultdict
import statistics

import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

try:
    import fitz  
except ImportError:
    fitz = None

DATA_DIR = "rag_data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
os.makedirs(DATA_DIR, exist_ok=True)

CHAT_MODEL_NAME = "models/gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

SYSTEM_PROMPT = (
    "You are TaskScapeAI, an assistant that answers questions ONLY\n"
    "using the content of the uploaded documents (syllabi, schedules, PDFs, etc.).\n\n"
    "Rules:\n"
    "1) Use only the provided document excerpts in the context.\n"
    "2) If the answer is not clearly present in the documents, say "
    "\"I don't know based on the documents.\"\n"
    "3) Do NOT invent dates, deadlines, courses, times, locations, or any schedule information.\n"
    "4) For schedule questions (e.g., \"What courses do I have on Monday?\"),\n"
    "   use schedule entries that explicitly have a day in their metadata.\n"
    "5) When you CAN clearly identify information (e.g., \"COE 414 meets M-W-F from 8:00–8:50\"),\n"
    "   state it concisely and in a structured way.\n"
    "6) Always be honest about uncertainty."
    "7) Check exam percentages and grading breakdowns if the user asks about how their final grade is calculated.\n"
    "8) Don't produce hateful, harmful, or biased content.\n"
    "9) Treat uploaded documents as confidential and do not share their contents.\n"
    "10) Whenever possible, quote the exact lines or phrases from the document chunks that support your answer.\n"

)


def _configure_google() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file at the project root."
        )
    genai.configure(api_key=api_key)


def embed_text(text: str) -> List[float]:
    _configure_google()
    emb = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=text,
    )
    return emb["embedding"]


class Gemini2EmbeddingFunction:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return embed_text(text)


def get_llm():
    _configure_google()
    model = genai.GenerativeModel(CHAT_MODEL_NAME)

    def chat(messages: List[dict]) -> str:
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            prefix = "User" if role == "user" else "System"
            prompt += f"{prefix}: {msg['content']}\n\n"
        response = model.generate_content(prompt)
        return response.text

    return chat



COURSE_RE = re.compile(r"^[A-Z]{3}\s+\d{3}[A-Z]?$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$")
LOCATION_RE = re.compile(r"^[A-Z][A-Za-z .]*\s+\d{3,4}$")

DAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

DAY_MAP = {
    "M": "monday",
    "T": "tuesday",
    "W": "wednesday",
    "R": "thursday",
    "F": "friday",
    "S": "saturday",
    "U": "sunday",
}



def _parse_schedule_column_text(col_text: str) -> List[Dict[str, str]]:
    lines = [l.strip() for l in col_text.splitlines() if l.strip()]
    entries: List[Dict[str, str]] = []
    i = 0
    pending_loc: Optional[str] = None

    while i < len(lines):
        line = lines[i]
        if COURSE_RE.match(line):
            course = line
            i += 1

            time_str = None
            if i < len(lines) and TIME_RE.match(lines[i]):
                time_str = lines[i]
                i += 1

            location = None
            if pending_loc:
                location = pending_loc
                pending_loc = None

            if i < len(lines) and LOCATION_RE.match(lines[i]):
                location = lines[i]
                i += 1

            entries.append(
                {
                    "course": course,
                    "time": time_str or "Time not specified",
                    "location": location or "Location not specified",
                }
            )
        else:
            if LOCATION_RE.match(line):
                pending_loc = line
            i += 1

    return entries





def parse_schedule_pdf(path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Parse a weekly schedule PDF using word-level positions:

    1) Find x-center of each day header (Monday...Saturday) from words.
    2) For every other word on the page (excluding left time column),
       assign it to the day whose header x-center is closest.
    3) For each day, cluster words into rows by y, build text lines,
       and run _parse_schedule_column_text on those lines.

    """
    if fitz is None:
        return {}

    doc = fitz.open(path)
    if doc.page_count == 0:
        return {}

    page = doc[0]

    words = page.get_text("words")

    header_x: Dict[str, float] = {}
    for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
        token = text.strip(",: ").lower()
        if token in DAY_NAMES and token not in header_x:
            header_x[token] = (x0 + x1) / 2.0

    if not header_x:
        return {}

    def nearest_day(x_center: float) -> str:
        best_day = None
        best_dist = None
        for day, hx in header_x.items():
            d = abs(x_center - hx)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_day = day
        return best_day or "monday"

    day_words: Dict[str, List[tuple]] = defaultdict(list)

    for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
        raw = text.strip()
        if not raw:
            continue

        token = raw.strip(",: ").lower()
        if token in DAY_NAMES:
            continue
        if x0 < 80:
            continue

        x_center = (x0 + x1) / 2.0
        day = nearest_day(x_center)
        day_words[day].append((x0, y0, raw))

    schedule: Dict[str, List[Dict[str, str]]] = {}

    for day, wlist in day_words.items():
        rows: Dict[int, List[tuple]] = defaultdict(list)
        for x0, y0, raw in wlist:
            key = round(y0 / 10) * 10
            rows[key].append((x0, raw))

        lines: List[str] = []
        for y in sorted(rows.keys()):
            line_text = " ".join(
                word for _, word in sorted(rows[y], key=lambda z: z[0])
            )
            lines.append(line_text)

        col_text = "\n".join(lines)
        entries = _parse_schedule_column_text(col_text)
        if entries:
            schedule[day] = entries

    return schedule



def extract_syllabus_schedule(text: str, filename: str) -> List[Document]:
    """
    Extract meeting times from syllabus-like PDFs using regex.

    Returns:
      - one summary Document
      - one Document per (day, time) entry with day metadata
    """
    docs: List[Document] = []

    if "syllabus" not in filename.lower() and "course" not in filename.lower():
        return docs

    course_match = re.search(r"\b([A-Z]{3}\s+\d{3}[A-Z]?)\b", text)
    course_code = course_match.group(1) if course_match else "Unknown Course"

    meeting_pattern = re.compile(
        r"([MTWRFSU](?:-[MTWRFSU])*)\s+(?:from\s+)?"
        r"(\d{1,2}:\d{2}\s*[AP]M)\s*[-\u2013]\s*(\d{1,2}:\d{2}\s*[AP]M)",
        re.IGNORECASE,
    )

    matches = list(meeting_pattern.finditer(text))
    if not matches:
        return docs

    schedule_lines: List[str] = []
    entry_docs: List[Document] = []

    for m in matches:
        days_abbrev = m.group(1)
        start_time = m.group(2)
        end_time = m.group(3)

        day_tokens = days_abbrev.split("-")
        days = [DAY_MAP[d.upper()] for d in day_tokens if d.upper() in DAY_MAP]
        if not days:
            continue

        schedule_lines.append(f"Course: {course_code}")
        schedule_lines.append("Days: " + ", ".join(d.capitalize() for d in days))
        schedule_lines.append(f"Time: {start_time} - {end_time}")
        schedule_lines.append("")

        for day in days:
            entry_docs.append(
                Document(
                    page_content=(
                        f"{course_code} on {day.capitalize()} "
                        f"{start_time} - {end_time} (from syllabus {filename})"
                    ),
                    metadata={
                        "source": filename,
                        "type": "syllabus_schedule_entry",
                        "day": day,
                        "course": course_code,
                        "time": f"{start_time} - {end_time}",
                        "location": "Location not specified in syllabus",
                    },
                )
            )

    summary_text = (
        "=== SYLLABUS SCHEDULE INFO ===\nFrom: " + filename + "\n\n" +
        "\n".join(schedule_lines)
    )
    summary_doc = Document(
        page_content=summary_text,
        metadata={"source": filename, "type": "syllabus_schedule"},
    )

    docs.append(summary_doc)
    docs.extend(entry_docs)
    return docs



def is_schedule_file(filename: str) -> bool:
    name = filename.lower()
    return any(kw in name for kw in ["schedule", "timetable", "weekly"]) and name.endswith(
        ".pdf"
    )


def load_file_to_docs(uploaded_file) -> List[Document]:
    """
    Save the uploaded file, load as Documents, and, if it's a schedule PDF,
    add structured per-day entries from the layout.
    """
    filename = uploaded_file.name
    path = os.path.join(DATA_DIR, filename)

    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")

    docs = loader.load()
    all_text = "\n".join(d.page_content for d in docs)

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = filename

    extra_docs: List[Document] = []

    if is_schedule_file(filename):
        schedule = parse_schedule_pdf(path)
        if schedule:
            lines: List[str] = [
                "=== WEEKLY SCHEDULE (parsed from layout) ===",
                f"Source file: {filename}",
                "",
            ]
            day_order = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            for day in day_order:
                if day in schedule and schedule[day]:
                    lines.append(day.capitalize() + ":")
                    for entry in schedule[day]:
                        lines.append(f"  Course: {entry['course']}")
                        lines.append(f"  Time: {entry['time']}")
                        lines.append(f"  Location: {entry['location']}")
                        lines.append("")
            summary_doc = Document(
                page_content="\n".join(lines),
                metadata={"source": filename, "type": "schedule"},
            )
            extra_docs.append(summary_doc)

            for day, entries in schedule.items():
                for entry in entries:
                    entry_text = (
                        f"{entry['course']} on {day.capitalize()} "
                        f"{entry['time']} in {entry['location']}"
                    )
                    extra_docs.append(
                        Document(
                            page_content=entry_text,
                            metadata={
                                "source": filename,
                                "type": "schedule_entry",
                                "day": day,
                                "course": entry["course"],
                                "time": entry["time"],
                                "location": entry["location"],
                            },
                        )
                    )

    syllabus_docs = extract_syllabus_schedule(all_text, filename)
    extra_docs.extend(syllabus_docs)

    return docs + extra_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split normal docs into chunks; keep schedule-related docs intact.
    """
    special_types = {
        "schedule",
        "schedule_entry",
        "syllabus_schedule",
        "syllabus_schedule_entry",
    }

    special_docs = [d for d in docs if d.metadata.get("type") in special_types]
    other_docs = [d for d in docs if d.metadata.get("type") not in special_types]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(other_docs)
    split_docs.extend(special_docs)
    return split_docs




_vectordb_cache: Optional[Chroma] = None


def close_vector_store() -> None:
    global _vectordb_cache
    if _vectordb_cache is not None:
        try:
            if hasattr(_vectordb_cache, "_client"):
                client = _vectordb_cache._client
                if hasattr(client, "reset"):
                    client.reset()
                if hasattr(client, "_system") and hasattr(client._system, "stop"):
                    client._system.stop()
                del _vectordb_cache._client
        except Exception:
            pass
        _vectordb_cache = None


def safe_remove_directory(path: str, max_retries: int = 3) -> bool:
    import time
    import gc
    from shutil import rmtree

    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                rmtree(path)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                close_vector_store()
                time.sleep(0.5)
                gc.collect()
            else:
                try:
                    backup_path = path + "old" + str(int(time.time()))
                    os.rename(path, backup_path)
                    return True
                except Exception:
                    return False
    return False


def _load_vector_store() -> Optional[Chroma]:
    global _vectordb_cache

    if not os.path.exists(CHROMA_DIR):
        _vectordb_cache = None
        return None

    if _vectordb_cache is not None:
        return _vectordb_cache

    _vectordb_cache = Chroma(
        embedding_function=Gemini2EmbeddingFunction(),
        persist_directory=CHROMA_DIR,
        collection_name="taskscape_documents",
    )
    return _vectordb_cache




def ingest_documents(uploaded_files) -> int:
    import time
    import gc

    global CHROMA_DIR

    close_vector_store()
    time.sleep(0.3)
    gc.collect()

    all_docs: List[Document] = []

    for file in uploaded_files:
        docs = load_file_to_docs(file)
        if not docs:
            continue
        chunks = split_documents(docs)
        all_docs.extend(chunks)

    if not all_docs:
        safe_remove_directory(CHROMA_DIR)
        return 0

    if not safe_remove_directory(CHROMA_DIR):
        old_chroma = CHROMA_DIR
        CHROMA_DIR = os.path.join(DATA_DIR, f"chroma_db_{int(time.time())}")
        print(f"Could not fully remove {old_chroma}, using {CHROMA_DIR} instead.")

    embedding_fn = Gemini2EmbeddingFunction()

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_name="taskscape_documents",
    )
    vectordb.persist()

    global _vectordb_cache
    _vectordb_cache = vectordb

    print("Extracting schedule blocks for scheduler...")
    schedule_blocks = extract_schedule_blocks()
    
    if schedule_blocks:
        save_schedule_blocks(schedule_blocks)
        num_days = len(schedule_blocks)
        total_blocks = sum(len(blocks) for blocks in schedule_blocks.values())
        print(f"✓ Extracted {total_blocks} schedule blocks across {num_days} days")
    else:
        print("No schedule blocks found in uploaded documents")
    
    return len(all_docs)


SCHEDULE_BLOCKS_FILE = os.path.join(DATA_DIR, "schedule_blocks.json")

def extract_schedule_blocks() -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract all blocked time slots from ingested schedules.
    Returns a dictionary mapping day names to lists of blocked time blocks.
    
    Format:
    {
        "monday": [
            {
                "start_time": "10:00",
                "end_time": "12:29",
                "course": "HOM 250",
                "location": "AKSOB 1407"
            },
            ...
        ],
        ...
    }
    """
    vectordb = _load_vector_store()
    if vectordb is None:
        return {}
    
    try:
        docs = vectordb.similarity_search(
            "weekly schedule classes courses",
            k=50  
        )
        
        schedule_entries = [
            d for d in docs 
            if d.metadata.get("type") in ["schedule_entry", "syllabus_schedule_entry"]
        ]
        
        if not schedule_entries:
            return {}
        
        schedule_blocks = {day: [] for day in DAY_NAMES}
        
        for entry in schedule_entries:
            day = entry.metadata.get("day")
            time_str = entry.metadata.get("time", "")
            course = entry.metadata.get("course", "Class")
            location = entry.metadata.get("location", "")
            
            if not day or not time_str:
                continue
            
            time_str = time_str.replace(" ", "")
            
            if "-" in time_str or "–" in time_str:
                separator = "-" if "-" in time_str else "–"
                parts = time_str.split(separator)
                
                if len(parts) == 2:
                    start = parts[0].strip()
                    end = parts[1].strip()
                    
                    start = _convert_to_24h(start)
                    end = _convert_to_24h(end)
                    
                    if start and end:
                        schedule_blocks[day].append({
                            "start_time": start,
                            "end_time": end,
                            "course": course,
                            "location": location
                        })
        
        schedule_blocks = {day: blocks for day, blocks in schedule_blocks.items() if blocks}
        
        return schedule_blocks
        
    except Exception as e:
        print(f"Error extracting schedule blocks: {e}")
        return {}

def _convert_to_24h(time_str: str) -> Optional[str]:
    """Convert time string to 24-hour format HH:MM"""
    time_str = time_str.strip().upper()
    
    if "AM" not in time_str and "PM" not in time_str:
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                hour = int(parts[0])
                minute = int(parts[1])
                if 0 <= hour < 24 and 0 <= minute < 60:
                    return f"{hour:02d}:{minute:02d}"
        except:
            pass
        return None
    
    try:
        is_pm = "PM" in time_str
        time_str = time_str.replace("AM", "").replace("PM", "").strip()
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        
        if is_pm and hour != 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0
        
        return f"{hour:02d}:{minute:02d}"
    except:
        return None


def save_schedule_blocks(schedule_blocks: Dict[str, List[Dict[str, Any]]]) -> None:
    """Save schedule blocks to JSON file."""
    try:
        with open(SCHEDULE_BLOCKS_FILE, 'w') as f:
            json.dump(schedule_blocks, f, indent=2)
        print(f"Saved schedule blocks to {SCHEDULE_BLOCKS_FILE}")
    except Exception as e:
        print(f"Error saving schedule blocks: {e}")


def load_schedule_blocks() -> Dict[str, List[Dict[str, Any]]]:
    """Load schedule blocks from JSON file."""
    try:
        if os.path.exists(SCHEDULE_BLOCKS_FILE):
            with open(SCHEDULE_BLOCKS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading schedule blocks: {e}")
    return {}


def _extract_day_from_query(query: str) -> Optional[str]:
    q = query.lower()
    for day in DAY_NAMES:
        if day in q:
            return day
    return None


def answer_question(query: str) -> str:
    vectordb = _load_vector_store()
    if vectordb is None:
        return "No documents ingested yet."

    lower_q = query.lower()
    schedule_keywords = [
        "schedule",
        "class",
        "classes",
        "course",
        "courses",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "when do i have",
        "what do i have",
        "what time",
    ]
    is_schedule_query = any(k in lower_q for k in schedule_keywords)
    day = _extract_day_from_query(query)

    if is_schedule_query and day:
        search_query = f"classes on {day}"
        docs = vectordb.similarity_search(
            search_query,
            k=20,
            filter={"day": day},
        )
    else:
        k = 10 if is_schedule_query else 5
        docs = vectordb.similarity_search(query, k=k)

    if not docs:
        return "I couldn't find anything relevant in your uploaded documents."

    context_parts: List[str] = []
    for i, d in enumerate(docs):
        source = d.metadata.get("source", "Unknown")
        doc_type = d.metadata.get("type", "document")
        context_parts.append(
            f"[DOCUMENT {i+1} - {source} ({doc_type})]\n{d.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User question:\n{query}\n\n"
        f"Relevant excerpts from your documents:\n{context}\n\n"
        "Answer the question using ONLY the information above.\n"
        "For day-specific schedule questions like \"What courses do I have on Monday?\",\n"
        "list each course in this format:\n\n"
        "COURSE CODE\n"
        "hh:mm-hh:mm\n"
        "LOCATION\n\n"
        "If there are multiple sections of the same course, list each block separately.\n"
        "If the documents do not clearly contain the answer, say:\n"
        "\"I don't know based on the documents.\""
    )

    llm = get_llm()
    result = llm([{"role": "user", "content": prompt}])
    return result