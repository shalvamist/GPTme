from GPTme.config import MODEL_ID, SYSTEM_PROMPT
from langchain_core.prompts import PromptTemplate

start_token_prompt = ""
end_token_prompt = ""
system_prompt = SYSTEM_PROMPT

if "mistral" in MODEL_ID:
    start_token_prompt = "<s>[INST]"
    end_token_prompt = "[/INST]"

if "llama" in MODEL_ID:
    start_token_prompt = ""
    end_token_prompt = ""

template = start_token_prompt + """
System: """ + system_prompt + """
Context: {context}
Question: {question}
Helpful Answer: Let's work this out in a step by step way to be sure we have the right answer.""" + end_token_prompt

if ("mistral" in MODEL_ID) and ("-code-" in MODEL_ID):
    template = """<|im_start|>system""" + SYSTEM_PROMPT + """<|im_end|><|im_start|>user{question}<|im_end|>""" + """
    <|im_start|> Context: {context} <|im_end|><|im_start|>assistant"""

PROMPT = PromptTemplate.from_template(template=template)
