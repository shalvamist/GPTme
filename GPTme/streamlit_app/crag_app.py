import streamlit as st

from GPTme.pipes.local_crag import run_crag_app
from GPTme.ingest.database import get_docs_list, create_db, clear_db
from GPTme.config import SOURCES_PATH, DB_PATH
import os
import shutil

current_file_list = get_docs_list()
state = 'Pending DB'
database_ready = False

def reset_db():
    global state
    global database_ready
    database_ready = False
    clear_db()
    state = 'Rebuilding DB - Please hold'
    create_db()
    state = 'Ready for you questions'
    database_ready = True

with st.sidebar:
    st.title("Local Corrective RAG with your Data")
    "[View the source code](https://github.com/shalvamist/GPTme/blob/main/GPTme/streamlit_app/crag_app.py)"
    # Loading the file
    uploaded_files = st.file_uploader("Upload an article", 
                                    accept_multiple_files=True,
                                    type=('txt', 'pdf', 'doc', 'docx', 'py', 'xls', 'csv'),
                                    )
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(SOURCES_PATH, uploaded_file.name)
            with open(save_path, mode='wb') as w:
                w.write(uploaded_file.getvalue())
            if os.path.isfile(save_path):
                print(f'File {uploaded_file.name} is successfully saved!')
        database_ready = False
        state = 'Rebuilding DB - Please hold'
        create_db()
        state = 'Ready for you questions'
        database_ready = True
    st.button("Reset Database", on_click=reset_db)


st.title("📝 files Q&A with Local CRAG")

st.write(f'### {state}')

# Capturing the question
question = st.text_input(
    "Ask something related to your docs",
    placeholder="Can you give me a short summary of the context?",
    # disabled=not uploaded_file,
)

if question and database_ready:
    response = run_crag_app(question=question)
    st.write("### Answer")
    st.write(response)