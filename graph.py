from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from typing import Dict, Any, TypedDict
import pprint
import re
import json
from langchain_community.tools import TavilySearchResults
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st


class GraphState(TypedDict):
    keys: Dict[str, Any]


def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("~-retrieve-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        return {"keys": {"documents": [], "question": question}}
        
    retriever = st.session_state.retriever
    # LangChain v0.3+ uses .invoke() instead of .get_relevant_documents()
    documents = retriever.invoke(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """Generate answer using LLM."""
    print("~-generate-~")
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    try:
        from prompts import rag_prompt
        from services import get_llm
        
        llm = get_llm()
        context = "\n\n".join(doc.page_content for doc in documents)

        # Create and run chain
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question} 
            | rag_prompt 
            | llm 
            | StrOutputParser()
        )

        generation = rag_chain.invoke({})

        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation
            }
        }

    except Exception as e:
        error_msg = f"Error in generate function: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {"keys": {"documents": documents, "question": question, 
                "generation": "Sorry, I encountered an error while generating the response."}}


def grade_documents(state):
    """Determines whether the retrieved documents are relevant."""
    print("~-check relevance-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    from prompts import grade_prompt
    from services import get_llm
    
    llm = get_llm()

    chain = (
        grade_prompt 
        | llm 
        | StrOutputParser()
    )

    filtered_docs = []
    search = "No"
    
    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                response = json_match.group()
            
            score = json.loads(response)
            
            if score.get("score") == "yes":
                print("~-grade: document relevant-~")
                filtered_docs.append(d)
            else:
                print("~-grade: document not relevant-~")
                search = "Yes"
                
        except Exception as e:
            print(f"Error grading document: {str(e)}")
            # On error, keep the document to be safe
            filtered_docs.append(d)
            continue

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}




def web_search(state):
    """Web search based on the re-phrased question using Tavily API."""
    print("~-web search-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    progress_placeholder.info("Initiating web search...")
    
    try:
        # Validate Tavily API key
        if not st.session_state.tavily_api_key:
            progress_placeholder.warning("Tavily API key not provided - skipping web search")
            return {"keys": {"documents": documents, "question": question}}
        
        progress_placeholder.info("Configuring search tool...")
        
        # Initialize Tavily search tool
        tool = TavilySearchResults(
            api_key=st.session_state.tavily_api_key,
            max_results=3,
            search_depth="advanced"
        )
        
        # Execute search with retry logic
        progress_placeholder.info("Executing search query...")
        try:
            search_results = execute_tavily_search(tool, question)
        except Exception as search_error:
            progress_placeholder.error(f"Search failed after retries: {str(search_error)}")
            return {"keys": {"documents": documents, "question": question}}
        
        if not search_results:
            progress_placeholder.warning("No search results found")
            return {"keys": {"documents": documents, "question": question}}
        
        # Process results
        progress_placeholder.info("Processing search results...")
        web_results = []
        for result in search_results:
            # Extract and format relevant information
            content = (
                f"Title: {result.get('title', 'No title')}\n"
                f"Content: {result.get('content', 'No content')}\n"
            )
            web_results.append(content)
        
        # Create document from results
        web_document = Document(
            page_content="\n\n".join(web_results),
            metadata={
                "source": "tavily_search",
                "query": question,
                "result_count": len(web_results)
            }
        )
        documents.append(web_document)
        
        progress_placeholder.success(f"Successfully added {len(web_results)} search results")
        
    except Exception as error:
        error_msg = f"Web search error: {str(error)}"
        print(error_msg)
        progress_placeholder.error(error_msg)
    finally:
        progress_placeholder.empty()
    return {"keys": {"documents": documents, "question": question}}


def decide_to_generate(state):
    print("~-decide to generate-~")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        print("~-decision: transform query and run web search-~")
        return "transform_query"
    else:
        print("~-decision: generate-~")
        return "generate"


def format_document(doc: Document) -> str:
    return f"""
    Source: {doc.metadata.get('source', 'Unknown')}
    Title: {doc.metadata.get('title', 'No title')}
    Content: {doc.page_content[:200]}...
    """


def format_state(state: dict) -> str:
    formatted = {}
    
    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value
            
    return formatted


def build_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Define the nodes by langgraph
    workflow.add_node("retrieve", retrieve) 
    workflow.add_node("grade_documents", grade_documents)  
    workflow.add_node("generate", generate) 
    workflow.add_node("transform_query", transform_query)  
    workflow.add_node("web_search", web_search) 

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

def transform_query(state):
    """Transform the query to produce a better question."""
    print("~-transform query-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    from prompts import transform_prompt
    from services import get_llm
    
    llm = get_llm()

    chain = transform_prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question}
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_tavily_search(tool, query):
    return tool.invoke({"query": query})
