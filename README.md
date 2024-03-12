# GTPme - Private conversations with your docs 

**GPTme** is totally free aiming to make local RAG & Corrective-RAG (CRAG) pipes as modular as possible

## Main features
1. Total privicy - your data doesn't leave your machine
2. LLMs infrastructure modularity - Currently supporting LlamaCPP + HF & Ollama wraped with Langchain :)
3. Work with the latest models HuggingFace & Ollama has to offer
4. Utilize LangGraph to run corrective RAG on your documents - Awesome blog post & video can be found here (https://blog.langchain.dev/agentic-rag-with-langgraph/)

## Install steps - Linux
1. Pull the repo & cd to GPTme folder
2. Create a conda env or a virtual env
3. Activate the env
4. Run 'source setup.sh'
5. Run 'streamlit run GPTme/streamlit_app/crag_app.py'

## Installation of LlamaCPP - included in the setup.sh script
I didn't have much luck with installing the package and getting the GPU to work, using the default Linux installation instructions i.e.
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```
So following the directions in this github [issue](https://github.com/abetlen/llama-cpp-python/issues/509) I generated the llama_cpp_setup.sh which worked every time! 
The main drawback is that it take ~5 min to build and install

## ToDo
1. Colab deployment of CRAG & RAG pipes
2. Enable more web UI paltforms - right now only StreamLit is supported
