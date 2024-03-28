from typing import Dict, TypedDict
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

from GPTme.ingest.database import get_retriever
from GPTme.config import OLLAMA_MODEL
from GPTme.llms import ollama_load_model, hf_load_model
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

### Nodes ###
def db_retrieve(state):
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
    retriever = get_retriever("similarity") # GPTme retriever
    documents.extend(retriever.get_relevant_documents(question))
    return {"keys": {"documents": documents, "question": question}}

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
    # llm = hf_load_model.mount_model(hf_load_model.download_model())
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

    # LLM
    llm = ollama_load_model.mount_model(format="json")

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords & related information to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out unrelevant info from our context. \n
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

    wrapper = DuckDuckGoSearchAPIWrapper(time="d", max_results=3)
    text = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
    docs = text.run(question)
    documents.append(Document(docs))
    news = DuckDuckGoSearchResults(api_wrapper=wrapper, source="new")
    docs = news.run(question)
    documents.append(Document(docs))

    return {"keys": {"documents": documents, "question": question}}

def wiki_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---WIKI SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    docs = wikipedia.run(question)
    documents.append(Document(docs))

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
            "generation":"After performing both MRR & Similarity RAGs I couldn't find relevant context. I checked online but could'nt find the answer.\nCan you please share more details in your question ?",
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
    filtered_documents = state_dict["documents"]
    # search = state_dict["similarity_search"]

    if len(filtered_documents) == 0:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
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
workflow.add_node("db_retrieve", db_retrieve)  # retrieve
workflow.add_node("web_search", web_search)  # web_search
workflow.add_node("wiki_search", wiki_search)  # web_search
workflow.add_node("transform_query", transform_query)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("terminate_search", terminate_search)  # terminate_search

# Build graph
workflow.set_entry_point("db_retrieve")
workflow.add_edge("db_retrieve", "transform_query")
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search","wiki_search")
workflow.add_edge("wiki_search", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "terminate_search": "terminate_search",
        "generate": "generate",
    },
)
workflow.add_edge("terminate_search", END)
workflow.add_edge("generate", END)

'''
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
'''

# Compile
app = workflow.compile()

def get_crag_app():
    """
    Implements the langgraph for local corrective RAG (CRAG). Graph -

    DB_retrieve (MMR, Similarity) --> Question transmute --> Websearch --> WiKiSearch --> grade_documents --> generate
                                                                                                |
                                                                                                V
                                                                                            Terminate


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
