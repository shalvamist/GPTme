from typing import Dict, TypedDict
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser

from GPTme.ingest.database import get_retriever
from GPTme.config import OLLAMA_MODEL
from GPTme.llms import ollama_load_model 

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

### Nodes ###
def mmr_retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    retriever = get_retriever() # GPTme retriever
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question, "search_type":"mmr"}}

def similarity_retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    retriever = get_retriever("similarity") # GPTme retriever
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question, "search_type":"similarity"}}

from langchain_community.chat_models import ChatOllama
from GPTme.prompt_config import PROMPT

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = PROMPT

    # LLM
    # llm = ChatOllama(model = OLLAMA_MODEL)
    llm = ollama_load_model.mount_model()

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    search_type = state_dict["search_type"]

    # LLM
    llm = ollama_load_model.mount_model(format="json")

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question", "context"],
    )

    chain = prompt | llm | JsonOutputParser()

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d.page_content,
            }
        )
        # print(d.page_content)
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    search = "No"  # Perform web search only if we have less than 3 results for context gen
    if len(filtered_docs) < 1:
        search = "Yes"  

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "similarity_search": search,
            "search_type": search_type,
        }
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question: """,
        input_variables=["question"],
    )

    # Grader
    # LLM
    llm = ollama_load_model.mount_model()

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question}
    }

def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "question": question}}

def terminate_search(state):
    """
    Ran both MRR & Similarity retrievers and didn't find relevant docs. updating the user

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---NO RELEVANT CONTEXT---")
    return {
        "keys": {
            "generation":"After performing both MRR & Similarity RAGs I couldn't find relevant context. can you please share more details in your question ?",
        }
    }

### Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE / SEARCH / TERMINATE---")
    state_dict = state["keys"]

    # question = state_dict["question"]
    # filtered_documents = state_dict["documents"]
    search = state_dict["similarity_search"]
    search_type = state_dict["search_type"]

    # print(f"current state - {state_dict}")

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        if search_type == "mmr":
            print("---DECISION: SIMILARITY SEARCH---")
            return "similarity_retrieve"
        else: 
            print("---DECISION: TERMINATE SEARCH---")
            return "terminate_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
import pprint

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("mmr_retrieve", mmr_retrieve)  # retrieve
workflow.add_node("similarity_retrieve", similarity_retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("terminate_search", terminate_search)  # terminate_search

# Build graph
workflow.set_entry_point("mmr_retrieve")
workflow.add_edge("mmr_retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "similarity_retrieve": "similarity_retrieve",
        "terminate_search": "terminate_search",
        "generate": "generate",
    },
)
workflow.add_edge("similarity_retrieve", "grade_documents")
workflow.add_edge("terminate_search", END)
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

def get_crag_app():
    """
    Implements the langgraph for local corrective RAG (CRAG). Graph -

                     terminate_search
                         ^
                         |
    mmr_retrieve --> grade_documents --> generate
                         |   ^
                         v   |
                     similarity_retrieve

    Args:
        inputs = {
            "keys": {
                "question": "what are the most important features of RISCV ?",
            }
        }

    Returns:
        str: the CRAG result 
    """   
    return app


def run_crag_app(question = "Hello"):
    inputs = {
        "keys": {
            "question": f"{question}",
        }
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    return value["keys"]["generation"]
