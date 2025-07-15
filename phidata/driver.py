from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
os.getenv('GOOGLE_API_KEY')

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[YFinanceTools(stock_price=True)],
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("Give me stock price of Nvida?")