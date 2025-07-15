# The imports

from dotenv import load_dotenv
import os
from agents import Agent, Runner, trace
import asyncio

# Load environment variables (make sure your .env has the Azure OpenAI settings)
load_dotenv(override=True)

# Set OpenAI Agents SDK to use Azure OpenAI endpoints
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")

# Use your Azure deployment/model name
agent = Agent(
    name="Jokester",
    instructions="You are a joke teller",
    model=os.getenv("AZURE_MODEL")  # e.g., "gpt-4o-mini"
)

async def main():
    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())