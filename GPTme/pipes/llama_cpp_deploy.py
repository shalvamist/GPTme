from llm.load_model import download_model, mount_model
from prompt_config import PROMPT
from langchain.chains import LLMChain

llm_path = download_model()
llm = mount_model(llm_path)

llm_chain = LLMChain(prompt=PROMPT, llm=llm)
question = "what is the history of Israel in the time of Napolion?"

print(llm_chain.run(question))