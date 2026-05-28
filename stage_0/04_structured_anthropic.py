from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()
client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"

# Tool build:
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_weather(city):
    return "Sunny"

TOOL_FUNCTIONS = {
    "get_time": get_time,
    "get_weather": get_weather,
}

TOOLS = [
    {
        "name": "get_time",
        "description": "Get the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    },
    {
        "name": "get_weather",
        "description": "Give the weather of the city which user asks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                }
            },
            "required": ["city"],
        }
    }
]

def run_tool_block(block):
    func = TOOL_FUNCTIONS[block.name]
    args = block.input
    result = func(**args)
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": str(result)
    }

def main():
    messages = [
        {
            "role": "user",
            "content": "How is the weather in Dallas and what time is it?"
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=TOOLS,
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")

        tool_output = run_tool_block(tool_use)

        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(tool_output)
                    }
                ],
            }
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
        )
    final_text = next(block for block in response.content if block.type == "text")
    print(final_text.text)
    return

if __name__ == "__main__":
    main()