from langchain_community.chat_models import ChatOllama
from GPTme.config import OLLAMA_MODEL, CONTEXT_WINDOW_SIZE, N_BATCH, LLM_TEMP, MAX_NEW_TOKENS

def mount_model(model_name = OLLAMA_MODEL, CtxN = CONTEXT_WINDOW_SIZE, BatchN = N_BATCH, temp = LLM_TEMP, format = "", newTokens = MAX_NEW_TOKENS):
    print(f"Using Langchin to mount {OLLAMA_MODEL} with Ollama")

    # for all function call options - https://github.com/langchain-ai/langchainjs/blob/12e5d43/libs/langchain-community/src/chat_models/ollama.ts#L109
    llm = ChatOllama(
        model= model_name,
        temperature=temp,
        numCtx = CtxN,
        numBatch = BatchN,
        format = format,
        num_predict = newTokens,
        )

    return llm    
