from agno.models.azure import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_default_llm():
    """
    Creates and returns a default AzureOpenAI instance using environment variables.

    Returns:
        AzureOpenAI: Configured AzureOpenAI instance.
    """
    return AzureOpenAI(
        id=os.getenv("AZURE_MODEL_NAME"),
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_MODEL_NAME"),
    )