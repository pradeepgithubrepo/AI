from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import random
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from typing import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
# Import the Azure OpenAI client from your utility helper
from utils.utility import Helperclass

# Load environment variables
load_dotenv(override=True)

# Initialize Azure OpenAI client using Helperclass
helper = Helperclass()
azure_llm = helper.openai_client()

serper = GoogleSerperAPIWrapper()

tool_search =Tool(
        name="search",
        func=serper.run,
        description="Useful for when you need more information from an online search"
    )

tools = [tool_search]

memory = MemorySaver()
# Step 1: Define the State object
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)

# Bind the tools to the graph builder
azure_llm_tools  = azure_llm.bind_tools(tools)

# graph_builder.add_tool(tool_search)
# Step 3: Create a Node

def chatbot(state: State):
    return {"messages": [azure_llm_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))


# # Step 3: Create a Node
graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")



# # Step 5: Compile the Graph
graph = graph_builder.compile(checkpointer=memory)

config = {
    "configurable": { "thread_id" : "1"}
}

def chat(user_input: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]},config =config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()