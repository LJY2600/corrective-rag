import os
from urllib.parse import urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_chroma import Chroma
import streamlit as st
import tempfile


def load_documents(file_or_url: str, is_url: bool = True) -> list:
    try:
        if is_url:
            # A .pdf URL must be parsed as a PDF; WebBaseLoader would run an HTML
            # parser over the binary body and embed decoded garbage.
            if urlparse(file_or_url).path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_or_url)
            else:
                loader = WebBaseLoader(file_or_url)
                loader.requests_per_second = 1
        else:
            file_extension = os.path.splitext(file_or_url)[1].lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_or_url)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []


def ingest_if_needed(docs, source_key):
    """Ingest documents into Chroma vectorstore if source changed."""
    if not docs or st.session_state.get("ingested_source") == source_key:
        return

    from config import CHUNK_SIZE, CHUNK_OVERLAP
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_splits = text_splitter.split_documents(docs)

    # Get embeddings and Chroma instance
    from services import get_embeddings, get_chroma
    embeddings = get_embeddings()
    vectorstore = get_chroma(embeddings)

    # Add documents to the vectorstore (Chroma auto-persists)
    vectorstore.add_documents(all_splits)
    retriever = vectorstore.as_retriever()
    st.session_state.ingested_source = source_key
    st.session_state.retriever = retriever

