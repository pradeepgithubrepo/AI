import os
import langchain
import google.cloud.firestore as firestore
from langchain_community.chat_message_histories import FirestoreChatMessageHistory
from langchain_openai import AzureChatOpenAI
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
    
    def load_history_from_firebase(self):
        """
        Retrieves the Firebase project ID from environment variables.
        
        Returns:
            str: The Firebase project ID.
        """
        firestore_client=  firestore.Client(project = "langchain-df390") 
        chat_history = FirestoreChatMessageHistory(
            firestore_client=firestore_client,
            collection_name="chat_history",
            session_id="user_session_id")
        return chat_history