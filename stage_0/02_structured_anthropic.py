import json
from typing import List

from anthropic import Anthropic
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL= "claude-haiku-4-5-20251001"

class Appointment(BaseModel):
    title: str
    date: str
    time: str
    attendees: List[str]

TOOLS = [
    {
        "name": "record_appointment",
        "description": "Extract the appoinment detail.",
        "input_schema": Appointment.model_json_schema(),
    }
]
def main():
    message = "Let's meet Tuesday at 3pm with Sarah and Tom about the budget"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": message
            }
        ],
        tools=TOOLS,
        tool_choice={"type": "tool", "name": "record_appointment", "disable_parallel_tool_use": True}
    )
    print(response.stop_reason, "\n")
    block_input = next(block for block in response.content if block.type == "tool_use")
    print(Appointment.model_validate(block_input.input))
    return

if __name__ == "__main__":
    main()