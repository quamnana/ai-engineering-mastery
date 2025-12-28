import os
import openai
import ollama
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "")


class EmbeddingFunction:
    def __init__(self, model_type="default"):
        self.model_type = model_type

        if model_type == "openai":
            self.embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key, model_name="text-embedding-3-small"
            )
        elif model_type == "ollama":
            self.embedding_func = embedding_functions.OllamaEmbeddingFunction(
                model_name="nomic-embed-text"
            )
        else:
            self.embedding_func = embedding_functions.DefaultEmbeddingFunction()


class LLM:
    def __init__(self, model_type="ollama"):
        self.model_type = model_type

        if self.model_type == "openai":
            self.client = openai.OpenAI(openai_api_key)
            self.model_name = "gpt-4o-mini"

        if self.model_type == "ollama":
            self.client = ollama
            self.model_name = "llama3.2"

    def generate_completions(self, messages):

        if self.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )

            return response.choices[0].message.content

        if self.model_type == "ollama":
            response = self.client.chat(model=self.model_name, messages=messages)
            return response["message"]["content"]


class RAGPipeline:
    def __init__(self, embedding_func, collection_name="space-facts"):
        client = chromadb.Client()

        # delete existing collection
        try:
            client.delete_collection(collection_name)
        except:
            pass

        # create new collection
        collection = client.create_collection(
            collection_name, embedding_function=embedding_func.embedding_func
        )

        # load and add new documents to collection
        documents = self.load_documents()
        collection.add(
            ids=[str(indx) for indx in range(len(documents))], documents=documents
        )

        print("Vector store setup successfully...")

        self.collection = collection

    def load_documents(self):
        df = pd.read_csv("./rag/space_facts.csv")
        documents = df["fact"].to_list()
        print("Successfully loaded documents")
        for indx, doc in enumerate(documents):
            print(f"{indx} - {doc}")

        return documents

    def find_related_chunks(self, query, top_k=2):
        results = self.collection.query(query_texts=[query], n_results=top_k)

        print("\nRelated chunks found:")
        for doc in results["documents"][0]:
            print(f"- {doc}")

        return list(
            zip(
                results["documents"][0],
                (
                    results["metadatas"][0]
                    if results["metadatas"][0]
                    else [{}] * len(results["documents"][0])
                ),
            )
        )

    def augment_prompt(self, query, related_chunks):
        context = "\n".join([chunk[0] for chunk in related_chunks])
        augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        print("\nAugmented prompt: ")
        print(augmented_prompt)

        return augmented_prompt

    def process_query(self, query, llm, top_k=2):
        print(f"\nProcessing query: {query}")

        # retrieve context that is relevant to the users query
        related_chunks = self.find_related_chunks(query, top_k)

        # augment both the user's query and relevant context, to be used as the user's prompt
        augmented_prompt = self.augment_prompt(query, related_chunks)

        response = llm.generate_completions(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who can answer questions about space but only answers questions that are directly related to the sources/documents given.",
                },
                {"role": "user", "content": augmented_prompt},
            ]
        )

        print("\nGenerated response:")
        print(response)

        references = [chunk[0] for chunk in related_chunks]
        return response, references


def select_models():
    # Select LLM Model
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama2")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    # Select Embedding Model
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embeddings")
    print("2. Chroma Default")
    print("3. Ollama Embeddings")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "ollama"}[choice]
            break
        print("Please enter 1, 2, or 3")

    return llm_type, embedding_type


def main():
    print("Starting the RAG pipeline demo...")

    # Select models
    llm_type, embedding_type = select_models()

    # Initialize models
    llm = LLM(llm_type)
    embedding_func = EmbeddingFunction(embedding_type)

    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"Using Embeddings: {embedding_type.upper()}")

    # Instantiate rag pipeline
    rag_pipeline = RAGPipeline(embedding_func)

    queries = [
        "What is the Hubble Space Telescope?",
        "Tell me about Mars exploration.",
    ]

    # Run queries

    for query in queries:
        print("\n" + "=" * 50)

        response, references = rag_pipeline.process_query(query=query, llm=llm)

        print("\nFinal Results:")
        print("-" * 30)
        print("Response:", response)
        print("\nReferences used:")
        for ref in references:
            print(f"- {ref}")
        print("=" * 50)


if __name__ == "__main__":
    main()
