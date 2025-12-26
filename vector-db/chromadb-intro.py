import chromadb

# create client
chroma_client = chromadb.Client()

# create collection
collection = chroma_client.get_or_create_collection(name="my_collection")


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
results = collection.query(query_texts=query)


for index, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][index]
    distance = results["distances"][0][index]

    print(
        f"Document {index} = id: {doc_id} - content: {document} - distance: {distance}"
    )
