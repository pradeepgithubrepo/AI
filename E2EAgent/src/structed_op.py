from typing import List
from pydantic import BaseModel, Field
from utils.utility import Helperclass
# Initialize Azure OpenAI client using Helperclass
helper = Helperclass()
azure_llm = helper.openai_client()

class MobileReview(BaseModel):
        phone_model: str = Field(description="Name and model of the phone")
        rating: float = Field(description="Overall rating out of 5")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")
        summary: str = Field(description="Brief summary of the review")

review_text = """
    Just got my hands on the new Galaxy S21 and wow, this thing is slick! The screen is gorgeous,
    colors pop like crazy. Camera's insane too, especially at night - my Insta game's never been
    stronger. Battery life's solid, lasts me all day no problem.
    Not gonna lie though, it's pretty pricey. And what's with ditching the charger? C'mon Samsung.
    Also, still getting used to the new button layout, keep hitting Bixby by mistake.
    Overall, I'd say it's a solid 4 out of 5. Great phone, but a few annoying quirks keep it from
    being perfect. If you're due for an upgrade, definitely worth checking out!
    """

structured_llm = azure_llm.with_structured_output(MobileReview)
output = structured_llm.invoke(review_text)
print(output)
print(output.pros)
