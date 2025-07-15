import asyncio
from utils.utility import Helperclass
from langchain_core.messages import HumanMessage, SystemMessage

print("Init firestore")

async def main():
    helperobj = Helperclass()
    client = helperobj.openai_client()
    # Use a unique session_id for each chat (could be user id, or a random string)
    chat_history = helperobj.load_history_from_firebase()
    system_msg = SystemMessage(
        content="You are a helpful assistant for a social media platform. You help users with their queries related to the platform's features, policies, and functionalities. Always provide concise and accurate information."
    )
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        chat_history.add_user_message(user_input)
        # Compose the message list with system prompt and history
        messages = [system_msg] + chat_history.messages
        resp = await asyncio.get_event_loop().run_in_executor(None, client.invoke, messages)
        print("Bot:", resp.content.strip())
        chat_history.add_ai_message(resp.content.strip())  # Store bot reply in Firebase

if __name__ == "__main__":
    asyncio.run(main())