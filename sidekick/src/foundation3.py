from openai import AsyncAzureOpenAI
from agents import set_default_openai_client,set_tracing_disabled
from dotenv import load_dotenv
import os
from agents import Agent, Runner, trace
import asyncio
from openai.types.chat import ChatCompletionMessageParam

# Load environment variables
load_dotenv()

# Create OpenAI client using Azure OpenAI
openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT")
)

# Set the default OpenAI client for the Agents SDK
set_default_openai_client(openai_client)
set_tracing_disabled(True)


# Create a banking assistant agent
joke_agent = Agent(
    name="Jokester",
    instructions="You are a joke teller",
    model=os.getenv("AZURE_MODEL")# A function tool defined elsewhere
)

async def main():

    result = await Runner.run(joke_agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())