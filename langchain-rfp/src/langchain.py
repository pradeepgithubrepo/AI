import os
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False
# Load environment variables from .env file
load_dotenv()

async def main():
    client = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        model=os.getenv("AZURE_MODEL"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
        openai_api_type="azure",
        temperature=0.0
    )
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        msg = HumanMessage(content=user_input)
        resp = await asyncio.get_event_loop().run_in_executor(None, client.invoke, [msg])
        print("Bot:", resp.content.strip())

if __name__ == "__main__":
    asyncio.run(main())