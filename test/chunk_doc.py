from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy
import json


load_dotenv();

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBEDED_MODEL = os.getenv("EMBEDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

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
                }
            )
    return documents;

# add embedding in chunks
def add_embeddings(chunks, batch_size=100):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        response = client.embeddings.create(
            model="nomic-embed-text",
            input=[chunk['text'] for chunk in batch]
        )
        for item in response.data:
            batch[item.index]['vector'] = item.embedding

    return chunks


def search(query, chunks, top_k=3):
    query_vector = get_embedding(query)

    results = []

    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk['vector'])
        results.append({
            "score":float(score),
            "source":chunk['source'],
            "text":chunk['text'],
            "chunk_id":chunk['chunk_id'],
            "size":len(chunk['text']),
        })

    results.sort(key=lambda rec:rec['score'], reverse=True)
    return results[:top_k]


def create_chunk():
    if not KB_FOLDER:
        raise RuntimeError("KB_FOLDER is not set")
    folder = Path(KB_FOLDER)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parents[1] / folder

    chunks = []
    documents = load_documents(folder)
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks;

def chunk_document(document):
    chunks = []
    chunk_id=0;
    for part in document["text"].split("\n\n"):
        part = part.strip();
        if part:
            chunks.append(
                {
                    "chunk_id":chunk_id,
                    "source":document["source"],
                    "text":part,
                }
            )
            chunk_id+=1;
    return chunks;


# print(str(documents));

def test(query, chunks):
    print(f"User Query : {query}")
    results = search(query, chunks, 3)
    for r in results:
        print(f" {r['score']:.2f} - {r['source']} - {r['size']} characters")
        print(r['text'])

def save_vector_store(chunks):
    file = Path(VECTOR_STORE_FILE)
    if not file.is_absolute():
        file = Path(__file__).resolve().parents[1] / file
    with open(file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)
def load_vector_store():
    file = Path(VECTOR_STORE_FILE)
    if not file.is_absolute():
        file = Path(__file__).resolve().parents[1] / file
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


# lets start testing
chunks = []
vectorFile = Path(VECTOR_STORE_FILE)
if not vectorFile.is_absolute():
    vectorFile = Path(__file__).resolve().parents[1] / vectorFile
if os.path.exists(vectorFile):
    chunks = load_vector_store()
    print(f"Loaded {len(chunks)} chunks from the vector store file {VECTOR_STORE_FILE}")
else:
    chunks = create_chunk()
    chunks = add_embeddings(chunks);
    save_vector_store(chunks)
    print(f"We have embedded {len(chunks)} chunks from our knowledge base")

user_query = "What are the benefits of Maternity Leave ?"

test(user_query, chunks)


