from agno.agent import Agent
# from agno.models.azure import AzureOpenAI
from agno.models.google import Gemini

from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
os.getenv('GOOGLE_API_KEY')


agent = Agent(
     model=Gemini(id="gemini-2.0-flash"),
    description="You are an enthusiastic news reporter with a flair for storytelling!.You also give me the source along with dates",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)
# agent.print_response("Write a report on NVDA", stream=True, show_full_reasoning=True, stream_intermediate_steps=True)
agent.print_response("Tell me about a breaking news story from Chennai.", stream=True)
# agent.print_response("Share a 2 sentence horror story.")