from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="Write a short bed time story about cats."
)

print(response.output_text)