from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

try:
    with client.responses.stream(
        model="gpt-4o",
        input="Write a bedtime story about bird."
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
        
        final_response = stream.get_final_response()
        print("\n\nFull text:")
        print(final_response.output_text)
except Exception as e:
    print("\n\nStreaming faild:")
    print(e)

