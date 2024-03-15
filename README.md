# GTPme - Private conversations with your docs 

**GPTme** is totally free aiming to make local RAG pipes as easy to integrate AND as modular as possible!

## Main features
1. Total privicy & security - your data doesn't leave your machine! 
2. LLMs infrastructure modularity - Currently supporting LlamaCPP + HF & Ollama wraped with Langchain - working on the non free services intergration right now
3. Work with the latest models HuggingFace 
4. Utilize LangGraph to run corrective RAG on your documents - Awesome blog post & videos can be found [here](https://blog.langchain.dev/agentic-rag-with-langgraph/)

## Install steps - Colab
One click set up ! 
1. Simple RAG example - checkout this [notebook](https://github.com/shalvamist/GPTme/blob/main/Examples/notebooks/GPTme_Colab.ipynb)   

## Install steps - Linux
1. Pull the repo & cd to GPTme folder
2. Create a virtual env (or conda env)
   ```bash
   python venv -m GPTme_env
   source GPTme_env/bin/activate
   ```
4. Run the setup script for the repo - this will - install required pip packages, build and install GPTme, install LlamaCPP with GPU support -
   ```bash
   source setup.sh
   ```

## Examples - once GPTme is setup
1. To run the streamlit app for local discussion with your docs run -
   ```bash
   streamlit run GPTme/streamlit_app/crag_app.py
   ```
2. Corrective RAG example - checkout this [script](https://github.com/shalvamist/GPTme/blob/main/Examples/crag_example.py) -
   ```bash
   python GPTme/Examples/crag_example.py
   ```

## Installation of LlamaCPP - included in the setup.sh script
I didn't have much luck with installing the package and getting the GPU to work, using the default Linux installation instructions i.e.
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```
So following the directions in this github [issue](https://github.com/abetlen/llama-cpp-python/issues/509) I created the llama_cpp_setup.sh which worked every time!

The main drawback is that it takes ~5 min to build and install - checkout the [notebook](https://github.com/shalvamist/GPTme/blob/main/Examples/notebooks/Running_LlamaCPP_in_Colab_GPU.ipynb) for more experimentation 

