# Example: Creating an agent with all attributes
from crewai import Agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from load_model import download_model, mount_model
from doc_loader import get_retriever, get_docs
from prompt_config import PROMPT
from langchain_core.runnables import RunnableParallel

llm_path = download_model()
llm = mount_model(llm_path)

agent = Agent(
  role='Data Analyst',
  goal='Extract actionable insights',
  backstory="""You're a data analyst at a large company.
  You're responsible for analyzing data and providing insights
  to the business.
  You're currently working on a project to analyze the
  performance of our marketing campaigns.""",
  tools=[my_tool1, my_tool2],
  llm=llm,
  function_calling_llm=llm,
  max_iter=10,
  max_rpm=10,
  verbose=True,
  allow_delegation=True,
  step_callback=my_intermediate_step_callback
)