from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
import os
import shutil

from GPTme.config import EMBEDDING_MODEL, DB_PATH, CHUNK_SIZE, OVERLAP, SOURCES_PATH, DEVICE
from GPTme.ingest.load_webpage import load_webpage

import logging
import os
from GPTme.config import SOURCES_PATH, CHUNK_SIZE, OVERLAP
from langchain_community.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Setting the embedding DB
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# TODO - review PDF loaders - need to find the best option
# Loading images -  
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    #".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

def file_log(logentry):
   file1 = open("file_ingest.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
       file_extension = os.path.splitext(file_path)[1]
       loader_class = DOCUMENT_MAP.get(file_extension)
       if loader_class:
           if file_extension == 'pdf':
              file_log(file_path + ' loaded.')
              loader = loader_class(file_path, mode="elements")
           else:
              file_log(file_path + ' loaded.')
              loader = loader_class(file_path)
       else:
           file_log(file_path + ' document type is undefined.')
           raise ValueError("Document type is undefined")
       return loader.load()[0]
    except Exception as ex:
       file_log('%s loading error: \n%s' % (file_path, ex))
       return None

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
           file_log(name + ' failed to submit')
           return None
        else:
           data_list = [future.result() for future in futures]
           # return data and file paths
           return (data_list, filepaths)

def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == ".py":
               python_docs.append(doc)
           else:
               text_docs.append(doc)
    return text_docs, python_docs

def load_documents(source_dir: str):
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    doc_names = []

    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print('Importing: ' + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            doc_names.extend(source_file_path)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(os.cpu_count(), max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)

    docs = []

    if chunksize == 0:
        return docs

    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log('executor task failed: %s' % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log('Exception: %s' % (ex))
    return docs

class RAG_DB:

    # Instance of the data base
    DATABASE = None
    # List of files in the data base
    DOC_LIST = []
    # Docs content for search / refernces
    DOCS_CONTENT = []

    def __init__(self) -> None:
        # Creating a folder for the DB
        if not os.path.exists(DB_PATH):
            os.mkdir(DB_PATH)

        # Check if the docs folder exists 
        if not os.path.exists(SOURCES_PATH):
            os.mkdir(SOURCES_PATH) 

    def load_sources(self):
        # Load documents 
        logging.info(f"Loading documents from {SOURCES_PATH}")
        DOCS_CONTENT = load_documents(SOURCES_PATH)
        text_documents, python_documents = split_documents(DOCS_CONTENT)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
        python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)

        texts = text_splitter.split_documents(text_documents)
        texts.extend(python_splitter.split_documents(python_documents))
        logging.info(f"Loaded {len(DOCS_CONTENT)} documents from {SOURCES_PATH}")
        logging.info(f"Split into {len(texts)} chunks of text")

        return texts

    def update_doc_list(self,doc_list):
        RAG_DB.DOC_LIST = doc_list

    def update_database(self,db):
        RAG_DB.DATABASE = db
    
    def update_contects(self,doc_contents):
        RAG_DB.DOCS_CONTENT = doc_contents

def create_db():
    ## There are a few cases to handle - 
    ## 1. First time creation - DATABASE will be None & We need to list all the docusments 
    ##      a. corner case - We might not have any docs
    ## 2. Nothing was added to the list of docs - no need to generate embeddings for the same list - return the DATABASE instance
    ## 3. New docs were added or some docs were removed - need to recreate data base
    db = RAG_DB()

    # Check DB status
    logging.info(f"Checking the status of database at: {DB_PATH}")
    doc_names = []

    for root, _, files in os.walk(SOURCES_PATH):
        for file_name in files:
            source_file_path = os.path.join(root, file_name)
            doc_names.extend(source_file_path)

    # Checking if the list of source docs has changed 
    if db.DOC_LIST != doc_names:
        # Somthing changed - need to rerun DB creation
        if len(doc_names) == 0:
            logging.info(f"No documents fonund in source dir: {SOURCES_PATH}")
            logging.info(f"Clearing out current DB folder")
            shutil.rmtree(DB_PATH)
            db.update_database(None)
            db.update_doc_list([])
            db.update_contects([])
            return None, []
        db.update_doc_list(doc_names)
    else:
        # Nothing changed no need to rerun the DB creation
        return db.DATABASE    

    # Loading sources from source directory
    all_splits = db.load_sources()
    print(f"Building DB from sources - using embedding model {EMBEDDING_MODEL}")

    # Splitting the documents 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, add_start_index=True
    )
    # Loading webpages from webpage list
    db.DOCS_CONTENT.extend(load_webpage())
    # Splitting the docs
    all_splits.extend(text_splitter.split_documents(load_webpage()))

    # Creating the vector DB - Indexing
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    model_kwargs = {'device':DEVICE}
    encode_kwargs = {'normalize_embeddings': True}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs,      # Pass the model configuration options
        encode_kwargs=encode_kwargs     # Pass the encoding options
    )

    # Create the DB 
    db.update_database(Chroma.from_documents(
        all_splits, 
        embedding=embeddings,
        persist_directory=DB_PATH,
        client_settings=CHROMA_SETTINGS,
        ))
    
    return db.DATABASE  

def get_retriever(type="mmr"):
    db = create_db()

    if type == "mmr":
        return db.as_retriever(
            search_type="mmr",  
            search_kwargs={"k": 5},
        )
    else:
        return db.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.2}
        )

def get_docs():
    db = RAG_DB()
    return db.DOCS_CONTENT