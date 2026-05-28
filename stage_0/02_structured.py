from datetime import date, time

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from typing import List, Optional, Literal
import json

load_dotenv()

class Appointment(BaseModel):
    model_config = ConfigDict(extra='forbid')

    title: str = Field(min_length=1)
    day: Literal["Monday", "Tuesday", "Wendsday", "Thursday", "Friday", "Saturday", "Sunday"]
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    attendees: List[str]

client = OpenAI()

sentence = "Let's meet Tuesday at 3pm with Sarah and Tom about the budget"

response = client.responses.create(
    model="gpt-4o",
    input=[{
        "role": "user",
        "content": f"Extract appointment details from: {sentence}"
    }],
    tools=[{
        "type": "function",
        "name": "record_appointment",
        "parameters": Appointment.model_json_schema(),
        "strict": True
    }],
    tool_choice={"type": "function", "name": "record_appointment"}
)

tool_call = next(item for item in response.output if item.type == "function_call")

args = json.loads(tool_call.arguments)
print(args, "\n")
appt = Appointment.model_validate(args)

print(appt)