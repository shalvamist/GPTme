from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

import os
import shutil



from GPTme.config import EMBEDDING_MODEL, DB_PATH, CHUNK_SIZE, OVERLAP
from GPTme.ingest.load_webpage import load_webpage
from GPTme.ingest.load_sourcedocs import load_sources

# Setting the embedding DB
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

DOCS_CONTENT = []
DATABASE = None

def init_db():

    global DOCS_CONTENT
    global DATABASE 

    # Creating a folder for the DB
    if not os.path.exists(DB_PATH):
        os.mkdir(DB_PATH)
    else:
        if len(DOCS_CONTENT) == 0:
            shutil.rmtree(DB_PATH)

    # Loading sources from source directory
    all_splits, docs, db_exists = load_sources()
    # In the case no new files were added just return the globals
    if db_exists:
        print("no need to re-init the DB")
        return DATABASE, DOCS_CONTENT
    else:
        DOCS_CONTENT = docs

    print(f"Building DB from sources - using embedding model {EMBEDDING_MODEL}")

    # Splitting the documents 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, add_start_index=True
    )
    # Loading webpages from webpage list
    DOCS_CONTENT.extend(load_webpage())
    # Splitting the docs
    all_splits.extend(text_splitter.split_documents(load_webpage()))

    # Creating the vector DB - Indexing
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    model_kwargs = {'device':'cuda'}
    encode_kwargs = {'normalize_embeddings': True}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs,      # Pass the model configuration options
        encode_kwargs=encode_kwargs     # Pass the encoding options
    )

    # Create the DB 
    DATABASE = Chroma.from_documents(
        all_splits, 
        embedding=embeddings,
        persist_directory=DB_PATH,
        client_settings=CHROMA_SETTINGS,
        )
    
    return DATABASE, DOCS_CONTENT    

def get_retriever(type="mmr"):
    db, DOCS_CONTENT = init_db()

    if type == "mmr":
        return db.as_retriever(
            search_type="mmr",  
            search_kwargs={"k": 10},
        )
    else:
        return db.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.2}
        )

def get_docs():
    return DOCS_CONTENT