import json

from openai import OpenAI
from dotenv import load_dotenv

from datetime import datetime

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o"


# Tools
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_weather(city):
    return "Sunny"


TOOL_FUNCTIONS = {"get_time": get_time, "get_weather": get_weather}

TOOLS = [
    {
        "type": "function",
        "name": "get_time",
        "description": "Get the current date and time",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather of the city which user asks.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city."}
            },
            "required": ["city"],
        },
    },
]


def run_tool_calls(call):
    func = TOOL_FUNCTIONS[call.name]
    args = json.loads(call.arguments)
    result = func(**args)
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": str(result),
    }


def main():
    question = "Write me a haiku"

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": question,
            }
        ],
        tools=TOOLS,
    )

    for _ in range(5):
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            print(response.output_text)
            return

        tool_call_outputs = [run_tool_calls(call) for call in tool_calls]

        response = client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=tool_call_outputs,
            tools=TOOLS,
        )
    print("Hit turn limit")


if __name__ == "__main__":
    main()
