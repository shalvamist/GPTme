import streamlit as st
from GPTme.pipes.rag_chain import get_ragchain
from GPTme.config import SOURCES_PATH
import os

with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 File Q&A with Local CRAG")

# Loading the file
uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))

# Capturing the question
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    save_path = os.path.join(SOURCES_PATH, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    if os.path.isfile(save_path):
        print(f'File {uploaded_file.name} is successfully saved!')

    response = get_ragchain(question=question)
    st.write("### Answer")
    st.write(response)