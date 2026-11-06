from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy

load_dotenv();

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text
    )
    return response.data[0].embedding;

def cosine_similarity(A,B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod/ (norm_a * norm_b)

text1 = "Python is a programming language"
text2 = "Python is used to write AI related code snippets"
text3 = "I love eating Apples every morning"


# print(get_embedding('hello there how are you'))


emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

sim12 = cosine_similarity(emb1, emb2);
sim13 = cosine_similarity(emb1, emb3);

print(f"similarity b/w text1 and text2 {sim12:.2f}")
print(f"similarity b/w text1 and text3 {sim13:.2f}")
