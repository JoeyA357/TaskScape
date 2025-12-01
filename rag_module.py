import os
import re
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# =============
# Directories
# =============

DATA_DIR = "rag_data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SCHEDULE_CACHE_FILE = os.path.join(DATA_DIR, "schedule_cache.txt")
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

Your goal is to give accurate, document-grounded answers that help the user understand both their academic obligations and their available time.
""".strip()

# =======================
# Schedule Parser
# =======================

class ScheduleParser:
    """Parses schedule PDFs and extracts structured class information."""
    
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    @staticmethod
    def is_schedule_file(filename: str) -> bool:
        """Check if filename suggests it's a schedule."""
        schedule_keywords = ['schedule', 'timetable', 'calendar']
        return any(keyword in filename.lower() for keyword in schedule_keywords)
    
    @staticmethod
    def parse_schedule_text(text: str) -> Optional[Dict]:
        """
        Parse schedule text and extract structured class information.
        Uses a column-based approach to correctly map classes to days.
        """
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Find the header line with day names
        day_header_line = None
        day_positions = {}
        
        for i, line in enumerate(lines):
            # Check if this line contains day names
            days_found = []
            for day in ScheduleParser.DAYS:
                if day in line:
                    pos = line.find(day)
                    days_found.append((day, pos))
            
            # If we found multiple days in this line, it's likely the header
            if len(days_found) >= 3:
                day_header_line = i
                day_positions = {day: pos for day, pos in days_found}
                break
        
        if not day_header_line or not day_positions:
            # Fallback to simple parsing if no clear header found
            return ScheduleParser._simple_parse(text)
        
        # Sort days by their position in the line
        sorted_days = sorted(day_positions.items(), key=lambda x: x[1])
        
        # Create schedule dictionary
        schedule = {day: [] for day, _ in sorted_days}
        
        # Pattern to match course codes
        course_pattern = r'\b([A-Z]{3}\s+\d{3}[A-Z]?)\b'
        time_pattern = r'(\d{1,2}:\d{2})[-–](\d{1,2}:\d{2})'
        
        # Process lines after the header
        current_course_info = {}
        
        for i in range(day_header_line + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            
            # Find course codes in this line
            course_matches = list(re.finditer(course_pattern, line))
            
            for match in course_matches:
                course_code = match.group(1)
                course_start_pos = match.start()
                
                # Determine which day column this course belongs to
                assigned_day = None
                min_distance = float('inf')
                
                for day, day_pos in sorted_days:
                    distance = abs(course_start_pos - day_pos)
                    if distance < min_distance:
                        min_distance = distance
                        assigned_day = day
                
                if not assigned_day:
                    continue
                
                # Look for time and location near this course
                time_str = None
                location_str = None
                
                # Search in current and next few lines
                for j in range(i, min(i + 4, len(lines))):
                    search_line = lines[j]
                    
                    # Only look in the same column region (within 50 chars of course position)
                    relevant_section = search_line[max(0, course_start_pos-20):course_start_pos+50]
                    
                    if not time_str:
                        time_match = re.search(time_pattern, relevant_section)
                        if time_match:
                            time_str = f"{time_match.group(1)}-{time_match.group(2)}"
                    
                    if not location_str:
                        # Look for location (building + room number)
                        loc_match = re.search(r'([A-Z][A-Za-z\s\.]+\s+\d+)', relevant_section)
                        if loc_match:
                            location_str = loc_match.group(1).strip()
                
                if time_str:
                    # Check if this exact class already exists for this day
                    exists = any(
                        cls['course'] == course_code and cls['time'] == time_str 
                        for cls in schedule[assigned_day]
                    )
                    
                    if not exists:
                        class_info = {
                            'course': course_code,
                            'time': time_str,
                            'location': location_str or 'Location not specified'
                        }
                        schedule[assigned_day].append(class_info)
        
        # Remove empty days
        schedule = {day: classes for day, classes in schedule.items() if classes}
        
        return schedule if schedule else None
    
    @staticmethod
    def _simple_parse(text: str) -> Optional[Dict]:
        """Fallback simple parsing method."""
        lines = text.split('\n')
        schedule = {day: [] for day in ScheduleParser.DAYS}
        
        course_pattern = r'([A-Z]{3}\s+\d{3}[A-Z]?)'
        time_pattern = r'(\d{1,2}:\d{2}[-–]\d{1,2}:\d{2})'
        
        for i, line in enumerate(lines):
            # Look for explicit day mentions followed by course info
            for day in ScheduleParser.DAYS:
                if day in line:
                    # Search next few lines for courses
                    for j in range(i, min(i + 10, len(lines))):
                        search_line = lines[j]
                        course_match = re.search(course_pattern, search_line)
                        time_match = re.search(time_pattern, search_line)
                        
                        if course_match and time_match:
                            course_code = course_match.group(1)
                            time_str = time_match.group(0)
                            
                            class_info = {
                                'course': course_code,
                                'time': time_str,
                                'location': 'See schedule'
                            }
                            schedule[day].append(class_info)
        
        schedule = {day: classes for day, classes in schedule.items() if classes}
        return schedule if schedule else None
    
    @staticmethod
    def format_schedule_for_storage(schedule: Dict) -> str:
        """Format parsed schedule into a structured text format for RAG."""
        lines = ["=== WEEKLY SCHEDULE ===\n"]
        
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day in day_order:
            if day in schedule and schedule[day]:
                lines.append(f"\n{day.upper()}:")
                for cls in sorted(schedule[day], key=lambda x: x['time']):
                    lines.append(f"  - {cls['course']}")
                    lines.append(f"    Time: {cls['time']}")
                    lines.append(f"    Location: {cls['location']}")
        
        return "\n".join(lines)

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
        return [embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
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

    docs = loader.load()
    
    # Check if this is a schedule file and parse it
    if ScheduleParser.is_schedule_file(filename):
        all_text = "\n".join([doc.page_content for doc in docs])
        parsed_schedule = ScheduleParser.parse_schedule_text(all_text)
        
        if parsed_schedule:
            # Create a formatted schedule document
            formatted_schedule = ScheduleParser.format_schedule_for_storage(parsed_schedule)
            
            # Save to cache for quick access
            with open(SCHEDULE_CACHE_FILE, 'w') as f:
                f.write(formatted_schedule)
            
            # Create a new document with the structured schedule
            schedule_doc = Document(
                page_content=formatted_schedule,
                metadata={"source": filename, "type": "schedule"}
            )
            docs.append(schedule_doc)
    
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents, but keep schedule documents intact."""
    schedule_docs = [doc for doc in docs if doc.metadata.get("type") == "schedule"]
    other_docs = [doc for doc in docs if doc.metadata.get("type") != "schedule"]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    split_docs = splitter.split_documents(other_docs)
    
    # Add schedule documents without splitting
    split_docs.extend(schedule_docs)
    
    return split_docs

# Ingest documents
def ingest_documents(uploaded_files):
    import time
    import gc
    global CHROMA_DIR
    
    # Close any existing vector store connection
    close_vector_store()
    
    # Give Windows time to release file locks
    time.sleep(0.5)
    gc.collect()
    
    all_docs = []

    for file in uploaded_files:
        docs = load_file_to_docs(file)
        if not docs:
            continue

        chunks = split_documents(docs)
        all_docs.extend(chunks)

    if not all_docs:
        # Try to safely remove the directory
        safe_remove_directory(CHROMA_DIR)
        return 0

    # Safely remove existing database with retries
    if not safe_remove_directory(CHROMA_DIR):
        # If we still can't delete, use a new directory name
        old_chroma = CHROMA_DIR
        CHROMA_DIR = os.path.join(DATA_DIR, f"chroma_db_{int(time.time())}")

    embedding_fn = Gemini2EmbeddingFunction()

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_name="taskscape_documents",
    )

    vectordb.persist()
    
    # Update cache with new instance
    global _vectordb_cache
    _vectordb_cache = vectordb
    
    return len(all_docs)

# Global variable to cache vector store
_vectordb_cache = None

# Load vector store
def _load_vector_store():
    global _vectordb_cache
    
    if not os.path.exists(CHROMA_DIR):
        _vectordb_cache = None
        return None
    
    # Return cached instance if available
    if _vectordb_cache is not None:
        return _vectordb_cache
    
    _vectordb_cache = Chroma(
        embedding_function=Gemini2EmbeddingFunction(),
        persist_directory=CHROMA_DIR,
        collection_name="taskscape_documents",
    )
    return _vectordb_cache

def close_vector_store():
    """Close the vector store to release file locks."""
    global _vectordb_cache
    if _vectordb_cache is not None:
        try:
            # Try multiple cleanup approaches
            if hasattr(_vectordb_cache, '_client'):
                if hasattr(_vectordb_cache._client, 'reset'):
                    _vectordb_cache._client.reset()
                if hasattr(_vectordb_cache._client, '_system'):
                    if hasattr(_vectordb_cache._client._system, 'stop'):
                        _vectordb_cache._client._system.stop()
                del _vectordb_cache._client
            del _vectordb_cache
        except Exception:
            pass
        _vectordb_cache = None

def safe_remove_directory(path, max_retries=3):
    """Safely remove a directory with retries for Windows file locks."""
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
                time.sleep(1)
                gc.collect()
            else:
                # Last attempt - try to rename instead of delete
                try:
                    backup_path = path + "_old_" + str(int(time.time()))
                    os.rename(path, backup_path)
                    return True
                except:
                    return False
    return False

# Answer question with schedule awareness
def answer_question(query: str) -> str:
    vectordb = _load_vector_store()
    if vectordb is None:
        return "No documents ingested yet."

    # Check if query is schedule-related
    schedule_keywords = ['class', 'schedule', 'monday', 'tuesday', 'wednesday', 
                         'thursday', 'friday', 'saturday', 'sunday', 'today', 
                         'tomorrow', 'time', 'when', 'what time']
    
    is_schedule_query = any(keyword in query.lower() for keyword in schedule_keywords)
    
    # If schedule query and we have cached schedule, prioritize it
    if is_schedule_query and os.path.exists(SCHEDULE_CACHE_FILE):
        with open(SCHEDULE_CACHE_FILE, 'r') as f:
            schedule_content = f.read()
        
        # Still do similarity search for context
        docs = vectordb.similarity_search(query, k=5)
        
        # Prepend schedule to context
        context_parts = [f"[SCHEDULE]\n{schedule_content}"]
        context_parts.extend([f"[DOC {i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
        context = "\n\n".join(context_parts)
    else:
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