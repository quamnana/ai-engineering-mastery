import chromadb
import os
from dotenv import load_dotenv

from chromadb.utils import embedding_functions

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# default embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

# openai embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-3-small"
)

croma_client = chromadb.PersistentClient(path="./db/chroma_persist")

collection = croma_client.get_or_create_collection(
    "my_story",
    embedding_function=openai_ef,
)


documents = [
    {"id": "id1", "content": "This is a document about pineapple"},
    {"id": "id2", "content": "This is a document about orange"},
    {"id": "id3", "content": "This is a document about mango"},
]

query = "This is a document about pineapple"

ids = []
contents = []
for doc in documents:
    ids.append(doc["id"])
    contents.append(doc["content"])

# adding documents using upsert func.
collection.upsert(ids=ids, documents=contents)

# query collection
results = collection.query(query_texts=query, n_results=2)


for index, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][index]
    distance = results["distances"][0][index]

    print(
        f" For the query: {query}, \n Found similar document: {document} (ID: {doc_id}, Distance: {distance})"
    )
