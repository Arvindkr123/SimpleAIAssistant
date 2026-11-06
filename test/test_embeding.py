from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv();

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

response = client.embeddings.create(
    model="nomic-embed-text",
    input=["I am good boy", "this is a cat", "this is my dog"]
)

for em in response.data:
    print(len(em.embedding))
# print(len(response.data[0].embedding))
# print(len(response.data[1].embedding))