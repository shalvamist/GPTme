from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from GPTme.llms.hf_load_model import download_model, mount_model
from GPTme.ingest.database import get_retriever, get_docs
from GPTme.prompt_config import PROMPT
from langchain_core.runnables import RunnableParallel

llm_path = download_model()
llm = mount_model(llm_path)
docs = get_docs()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_ragchain():    

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | PROMPT
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": get_retriever(), "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

result = get_ragchain().invoke("tell me about Italy")
print(result["answer"])



