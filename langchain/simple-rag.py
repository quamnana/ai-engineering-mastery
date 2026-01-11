import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


# Data cleaning function
def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


# document loader
text_loader = TextLoader(file_path="./langchain/data/dream.txt")
document = text_loader.load()

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splitted_documents = text_splitter.split_documents(document)

# text clean up
cleaned_texts = [clean_text(doc.page_content) for doc in splitted_documents]


# use OpenAI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# set up a vector as a retriever
retriever = FAISS.from_texts(cleaned_texts, embedding=embeddings).as_retriever(
    search_kwarg={"k": 3}
)

# user's query
query = "what did Martin Luther King Jr. dream about?"

# using user's query to retrieve relevant documents
retrieved_documents = retriever.invoke(query)

print(f"Retrieved Docs: {retrieved_documents}")

# create a prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful AI chatbot, use the following context: {documents} to answer the question: {query}"
)

model = ChatOpenAI(model="gpt-4.1")

chain = prompt | model | StrOutputParser()

response = chain.invoke({"documents": retrieved_documents, "query": query})

print(f"\nAI Response: {response}")
