from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from config import WEB_URLS, EMBEDDING_MODEL, DB_PATH, CHUNK_SIZE, OVERLAP

from load_webpage import load_webpage
from load_sourcedocs import load_sources

# Setting the embedding DB
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Splitting the documents 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, add_start_index=True
)

# Loading sources from source directory
all_splits, docs = load_sources()
# Loading webpages from webpage list
docs.extend(load_webpage())
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

db = Chroma.from_documents(
    all_splits, 
    embedding=embeddings,
    persist_directory=DB_PATH,
    client_settings=CHROMA_SETTINGS,
    )

# Creating the retriver 
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 20},
)

def get_retriever():
    return retriever

def get_docs():
    return docs
# retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# print(retrieved_docs)