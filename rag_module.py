# rag_module.py

from typing import List
import streamlit as st

def ingest_documents(uploaded_files: List):
    """
    Takes uploaded Streamlit file objects and:
      - reads them
      - chunks & embeds
      - stores in vector DB
    """
    # TODO: implement RAG ingestion
    ...

def answer_question(query: str) -> str:
    """
    Uses the vector DB + LLM to answer queries about uploaded docs.
    """
    # TODO: implement RAG question answering
    ...
