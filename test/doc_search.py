from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy


load_dotenv();

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBEDED_MODEL = os.getenv("EMBEDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")

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

# LOADS DOCUMENTS
def load_documents(folder=KB_FOLDER):
    documents = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.txt'):
            continue

        path = os.path.join(folder, filename);
        with open(path, 'r', encoding='utf-8') as fs:
            text = fs.read()
            documents.append(
                {
                    "source":filename,
                    "text": text,
                    "vector":get_embedding(text)
                }
            )
    return documents;

def search(query, documents, top_k=3):
    query_vector = get_embedding(query)

    results = []

    for document in documents:
        score = cosine_similarity(query_vector, document['vector'])
        results.append({
            "score":float(score),
            "source":document['source'],
            "text":document['text'],
            "size":len(document['text']),
        })

    results.sort(key=lambda rec:rec['score'], reverse=True)
    return results[:top_k]


if not KB_FOLDER:
    raise RuntimeError("KB_FOLDER is not set")

folder = Path(KB_FOLDER)
if not folder.is_absolute():
    folder = Path(__file__).resolve().parents[1] / folder

documents = load_documents(folder)
# print(str(documents));

def test(query, documents):
    print(f"User Query : {query}")
    results = search(query, documents, 3)
    for r in results:
        print(f" {r['score']:.2f} - {r['source']} - {r['size']} characters")

test('"How many days I will get for maternity leave ?"', documents)


