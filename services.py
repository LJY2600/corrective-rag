from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from config import LLM_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION


def get_llm():
    """Get LLM instance (Qwen via DashScope)."""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=st.session_state.dashscope_api_key,
        temperature=0,
        max_tokens=1000
    )


def get_embeddings():
    """Get embedding model (DashScope text-embedding-v3)."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LLM_BASE_URL,
        api_key=st.session_state.dashscope_api_key,
        dimensions=EMBEDDING_DIMENSION,
        check_embedding_ctx_length=False, 
        chunk_size=10,
    )


def get_chroma(embeddings):
    """Get Chroma vectorstore (factory)."""
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

