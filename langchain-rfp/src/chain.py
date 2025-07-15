import asyncio
from utils.utility import Helperclass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableSequence

async def main():
    helperobj = Helperclass()
    client = helperobj.openai_client()

    prompt_template  = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant who knows about {animal}"),
        ("human", "Tell me {fact_count} facts")
    ])

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        chain = prompt_template | client | StrOutputParser()
        # Prepare the input for the chain

        result = chain.invoke({
            "animal": user_input,
            "fact_count": 3
        })
        print("Bot:", result.strip())

if __name__ == "__main__":
    asyncio.run(main())