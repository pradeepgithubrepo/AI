#This basic example demostrate the LLM response and ChatModel Response

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
import openai
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the .env file
load_dotenv(find_dotenv())

# Retrieve Azure OpenAI specific configuration from environment variables
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Set the OpenAI library configuration using the retrieved environment variables
openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

# Initialize an instance of AzureOpenAI using the specified settings
llm = AzureOpenAI(
    openai_api_version=OPENAI_API_VERSION,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    openai_api_type=OPENAI_API_TYPE,
    deployment_name="gpt-4o-mini"  # Name of the deployment for identification
)

# Initialize an instance of AzureChatOpenAI using the specified settings
chat_llm = AzureChatOpenAI(
    openai_api_version=OPENAI_API_VERSION,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    openai_api_type=OPENAI_API_TYPE,
    deployment_name="gpt-4o-mini"  
)

# Print the response from AzureOpenAI LLM for a specific question
print("AzureOpenAI LLM Response: ", llm(" what is the weather in mumbai today?"))

# Print the response from AzureChatOpenAI for the same question
print("AzureOpenAI ChatLLM Response: ", chat_llm.predict("what is the weather in mumbai today?"))