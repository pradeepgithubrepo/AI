import asyncio
from utils.utility import Helperclass
from langchain_core.messages import HumanMessage

def get_judgement_prompt(competitors, question, answers, judge_model_name="Gemini"):
    together = ""
    for index, answer in enumerate(answers):
        together += f"# Response from competitor {index+1}\n\n"
        together += answer + "\n\n"
    judge = f"""You are {judge_model_name}, acting as a judge for a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", ...], "explanation": "Short explanation of why you ranked them this way", "judge_model": "{judge_model_name}"}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, an explanation for your ranking, and your model name. Do not include markdown formatting or code blocks."""
    return judge

async def main():
    helper = Helperclass()
    print("Select model: [1] Azure OpenAI  [2] Gemini  [3] Both (compare)")
    model_choice = input("Enter 1, 2 or 3: ").strip()
    if model_choice == "1":
        client = helper.openai_client()
        is_gemini = False
        is_both = False
    elif model_choice == "2":
        client = helper.gemini_client()
        is_gemini = True
        is_both = False
    elif model_choice == "3":
        azure_client = helper.openai_client()
        gemini_client = helper.gemini_client()
        is_both = True
    else:
        print("Invalid selection.")
        return

    print("Chat (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        if is_both:
            # Query both models
            msg = HumanMessage(content=user_input)
            azure_resp = await asyncio.get_event_loop().run_in_executor(None, azure_client.invoke, [msg])
            gemini_resp = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input
            )
            responses = [
                {
                    "competitor": "Azure OpenAI",
                    "question": user_input,
                    "response": azure_resp.content.strip()
                },
                {
                    "competitor": "Gemini",
                    "question": user_input,
                    "response": gemini_resp.text
                }
            ]
            for item in responses:
                print(f"\n--- {item['competitor']} Response ---")
                print(f"Question: {item['question']}")
                print(f"Response: {item['response']}")
            print("\n" + "="*40 + "\n")
        elif is_gemini:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input
            )
            print("Bot:", response.text)
        else:
            msg = HumanMessage(content=user_input)
            resp = await asyncio.get_event_loop().run_in_executor(None, client.invoke, [msg])
            print("Bot:", resp.content.strip())

        # Judging logic (for both models)
        if is_both:
            competitors = ["Azure OpenAI", "Gemini"]
            question = user_input
            answers = [azure_resp.content.strip(), gemini_resp.text]

            judge_prompt = get_judgement_prompt(competitors, question, answers, judge_model_name="Azure OpenAI")
            # Use Azure OpenAI as the judge LLM
            judge_msg = HumanMessage(content=judge_prompt)
            judge_resp = await asyncio.get_event_loop().run_in_executor(
                None, azure_client.invoke, [judge_msg]
            )
            print("\n--- Judge (Azure OpenAI) Ranking & Explanation ---")
            print(judge_resp.content.strip())

if __name__ == "__main__":
    asyncio.run(main())