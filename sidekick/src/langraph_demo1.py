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

# Import the Azure OpenAI client from your utility helper
from utils.utility import Helperclass

# Load environment variables
load_dotenv(override=True)

# Initialize Azure OpenAI client using Helperclass
helper = Helperclass()
azure_llm = helper.openai_client()

# Step 1: Define the State object
class State(BaseModel):
    messages: Annotated[list, add_messages]

# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)


# Step 3: Create a Node

def chatbot_node(old_state: State) -> State:
    response = azure_llm.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state

graph_builder.add_node("chatbot", chatbot_node)

# Step 4: Create Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Step 5: Compile the Graph
graph = graph_builder.compile()

def chat(user_input: str, history):
    initial_state = State(messages=[{"role": "user", "content": user_input}])
    result = graph.invoke(initial_state)
    print(result)
    return result['messages'][-1].content


gr.ChatInterface(chat, type="messages").launch()