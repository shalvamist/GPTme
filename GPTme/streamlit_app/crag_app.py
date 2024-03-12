import streamlit as st

from GPTme.pipes.local_crag import run_crag_app
from GPTme.config import SOURCES_PATH
import os

with st.sidebar:
    st.title("Local Corrective RAG with your Data")
    "[View the source code](https://github.com/shalvamist/GPTme/blob/main/GPTme/streamlit_app/crag_app.py)"

st.title("📝 files Q&A with Local CRAG")

# Loading the file
uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))

# Capturing the question
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary of the context?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    save_path = os.path.join(SOURCES_PATH, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    if os.path.isfile(save_path):
        print(f'File {uploaded_file.name} is successfully saved!')

    response = run_crag_app(question=question)
    st.write("### Answer")
    st.write(response)