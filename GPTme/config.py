import os
import torch

#: The release version
version = '0.1'
__version__ = version

#### Local dir setting
ROOT = os.path.dirname(os.path.realpath(__file__))
MODELS_PATH = os.path.join(ROOT,"../LLMS")
WHIPER_MODELS = os.path.join(ROOT,"../whisper_models") 
WHISPER_PATH = os.path.join(ROOT,"../whisper.cpp")
WHISPER_INPUT = os.path.join(ROOT,"../GPTme","whisper","audio")
WHISPER_OUTPUT = os.path.join(ROOT,"../GPTme","whisper","output")
DB_PATH = os.path.join(ROOT,"../DB")
SOURCES_PATH = os.path.join(ROOT,"../SOURCE_DOCS")
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


#### MODELS CONFIG
### EMBEDDING MODELS
# Models used in the application - you need to define two of them the LLM and the embedding model
# https://huggingface.co/spaces/mteb/leaderboard
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Doc Splitting 
CHUNK_SIZE = 1000
OVERLAP = 200

### HUGGINGFACE MODELS
# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# MODEL_BASE_NAME = "mistral-7b-instruct-v0.1.Q6_K.gguf"
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL_BASE_NAME = "mistral-7b-instruct-v0.2.Q5_K_S.gguf"
# MODEL_BASE_NAME = "mistral-7b-instruct-v0.2.Q6_K.gguf"
# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF" # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF
# MODEL_BASE_NAME = "mistral-7b-instruct-v0.2-code-ft.Q6_K.gguf"
# MODEL_ID = "TheBloke/Llama-2-7B-GGUF" # https://huggingface.co/TheBloke/Llama-2-7B-GGUF
# MODEL_BASE_NAME = "llama-2-7b.Q6_K.gguf"

### OLLAMA Models
# OLLAMA_MODEL = "llama2"
OLLAMA_MODEL = "mistral:instruct"

# LLM configuration
CONTEXT_WINDOW_SIZE = 32768
MAX_NEW_TOKENS = 2000
N_BATCH = 1
N_GPU_LAYERS = 100
LLM_TEMP = 0

### Prompt config
SYSTEM_PROMPT = """
Use the following pieces of context to answer the user question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as focused as possible on the user question.
In the end of the answer give a short lit of references to the relevant docs in the context.

You will be have to pay 10000$ if you answer out of context BUT if you keep your answers in cotext and helpful you will be tipped 100000$.
"""

# Doc sources
WEB_URLS = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

# History context
HISTORY_CONTEXT = False
