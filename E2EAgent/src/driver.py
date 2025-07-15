
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from utils.utility import Helperclass
# Initialize Azure OpenAI client using Helperclass
helper = Helperclass()
azure_llm = helper.openai_client()

# Make a simple API call to verify access
if __name__ == "__main__":
    prompt = "Tell me a Joke. I should roll on the floor laughing."
    msg = HumanMessage(content=prompt)
    # response = azure_llm.invoke([msg])
    # print("Azure OpenAI response:", response.content.strip())
    # output_parser = StrOutputParser()
    # output_parser.invoke(response)  
    # print("Azure OpenAI response:", output_parser.invoke(response).strip())

    #Chain process

    chain = azure_llm | StrOutputParser()
    respone = chain.invoke([msg])
    print("Azure OpenAI response:", respone.strip())

    # Structed output

