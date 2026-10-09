import streamlit as st
from ingestion import load_documents, ingest_if_needed
from graph import build_graph
import nest_asyncio
import pprint

nest_asyncio.apply()


def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        # Initialize API keys and URLs
        st.session_state.dashscope_api_key = ""
        st.session_state.tavily_api_key = ""
        st.session_state.doc_url = "https://arxiv.org/pdf/2307.09288.pdf"  


def setup_sidebar():
    """Setup sidebar for API keys."""
    with st.sidebar:
        st.subheader("API Configuration")
        st.session_state.dashscope_api_key = st.text_input(
            "DashScope API Key (required)", 
            value=st.session_state.dashscope_api_key, 
            type="password", 
            help="Required for Qwen LLM via Alibaba Cloud"
        )
        st.session_state.tavily_api_key = st.text_input(
            "Tavily API Key (optional)", 
            value=st.session_state.tavily_api_key, 
            type="password",
            help="Used for web search. Leave empty to skip."
        )
        st.session_state.doc_url = st.text_input(
            "Document URL", 
            value=st.session_state.doc_url
        )

        # Validate required key
        if not st.session_state.dashscope_api_key:
            st.warning("DashScope API Key is required to run the app.")
            st.stop()

        st.session_state.initialized = True


# --- Main App Logic ---
initialize_session_state()
setup_sidebar()

st.title("🔄 Corrective RAG Agent")

st.text("A possible query: What are the experiment results and ablation studies in this research paper?")

# Document Input
st.subheader("Document Input")
input_option = st.radio("Choose input method:", ["URL", "File Upload"])

docs = None  
source_key = None

if input_option == "URL":
    url = st.text_input("Enter document URL:", value=st.session_state.doc_url)
    if url:
        docs = load_documents(url, is_url=True)
        source_key = url
else:
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'md'])
    if uploaded_file:
        import tempfile
        import os
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
        # Clean up the temporary file
        os.unlink(tmp_file.name)
        source_key = uploaded_file.name

# Ingest if needed
if docs:
    ingest_if_needed(docs, source_key)

# Build graph
app = build_graph()

# User input
user_question = st.text_input("Please enter your question:")

if user_question:
    inputs = {
        "keys": {
            "question": user_question,
        }
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            with st.expander(f"Step '{key}':"):
                from graph import format_state
                st.text(pprint.pformat(format_state(value["keys"]), indent=2, width=80))

    final_generation = value['keys'].get('generation', 'No final generation produced.')
    st.subheader("Final Generation:")
    st.write(final_generation)
