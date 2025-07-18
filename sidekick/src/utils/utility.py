import os
import langchain

from langchain_community.chat_message_histories import FirestoreChatMessageHistory
from langchain_openai import AzureChatOpenAI
from google import genai

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

class Helperclass:
    def __init__(self):
        pass

    def openai_client(self):
        client = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            model=os.getenv("AZURE_MODEL"),
            api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_API_KEY"),
            openai_api_type="azure",
            temperature=0.0
        )
        return client

    def gemini_client(self):
        """
        Creates and returns a Google Gemini client using the API key from environment variables.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=api_key)
