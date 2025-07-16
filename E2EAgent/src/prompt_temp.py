from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.utility import Helperclass
# Initialize Azure OpenAI client using Helperclass
helper = Helperclass()
azure_llm = helper.openai_client()

# prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
# output_parser = StrOutputParser()
# chain = prompt | azure_llm | output_parser
# result = chain.invoke({"topic": "tesla"})
# print(result)

## System message and Human message

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me a short joke about {topic}")
])
output_parser = StrOutputParser()
chain = prompt | azure_llm | output_parser      
result = chain.invoke({"topic": "pakisthan"})
print(result)