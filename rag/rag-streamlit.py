import os
import openai
import ollama
import chromadb
import streamlit as st
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
            self.client = openai.OpenAI(api_key=openai_api_key)
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
        return response, references, augmented_prompt


def streamlit_app():
    st.set_page_config(page_title="Space Facts RAG", layout="wide")
    st.title("🚀 Space Facts RAG System")

    # Sidebar for model selection
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama Llama2",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["openai", "chroma", "ollama"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "chroma": "Chroma Default",
            "ollama": "Ollama Embeddings",
        }[x],
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

        # Initialize models
        st.session_state.llm = LLM(llm_type)
        st.session_state.embedding_func = EmbeddingFunction(embedding_type)

        rag_pipeline = RAGPipeline(st.session_state.embedding_func)
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.facts = rag_pipeline.load_documents()

        st.session_state.initialized = True

    # If models changed, reinitialize
    if (
        st.session_state.llm.model_type != llm_type
        or st.session_state.embedding_func.model_type != embedding_type
    ):
        st.session_state.llm = LLM(llm_type)
        st.session_state.embedding_func = EmbeddingFunction(embedding_type)

        rag_pipeline = RAGPipeline(st.session_state.embedding_func)
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.facts = rag_pipeline.load_documents()

    # Display available facts
    with st.expander("📚 Available Space Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact}")

    # Query input
    query = st.text_input(
        "Enter your question about space:",
        placeholder="e.g., What is the Hubble Space Telescope?",
    )

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = (
                st.session_state.rag_pipeline.process_query(query, st.session_state.llm)
            )

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Response")
                st.write(response)

            with col2:
                st.markdown("### 📖 References Used")
                for ref in references:
                    st.write(f"- {ref}")

            # Show technical details in expander
            with st.expander("🔍 Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")


if __name__ == "__main__":
    streamlit_app()
