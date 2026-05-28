import json
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o"

# --- Tool Implementation ---

def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOL_FUNCTIONS = {
    "get_time": get_time,
}

TOOLS = [
    {
        "type": "function",
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
]

def run_tool_call(call):
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

    tool_calls = [item for item in response.output if item.type == "function_call"]

    if not tool_calls:
        print(response.output_text)
        return

    tool_outputs = [run_tool_call(call) for call in tool_calls]
    second_response = client.responses.create(
        model=MODEL,
        previous_response_id=response.id,
        input=tool_outputs,
    )
    print(second_response.output_text)

if __name__ == "__main__":
    main()