import os
import tempfile
import whisper
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

load_dotenv()


def load_documents(directory):
    """Load documents from different file types"""
    loaders = {
        ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
        ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
        ".md": DirectoryLoader(
            directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
        ),
    }

    documents = []
    for file_type, loader in loaders.items():
        try:
            documents.extend(loader.load())
            print(f"Loaded {file_type} documents")
        except Exception as e:
            print(f"Error loading {file_type} documents: {str(e)}")

    return documents


def preprocess_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
    )

    # split article text into chunks
    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vector_store(documents, persist_directory):
    """Create and persist vector store if it doesn't exist, otherwise load existing one"""
    embeddings = OpenAIEmbeddings()

    # Check if persist_directory exists and has content
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        # Load existing vector store
        vector_store = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
    else:
        print(f"Creating new vector store in {persist_directory}")
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Create new vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        vector_store.persist()

    return vector_store


def initialize_external_clients():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs = ElevenLabs(api_key=api_key)

    whisper_model = whisper.load_model("base")
    return elevenlabs, whisper_model


def generate_text_to_speech(elevenlabs, text):
    try:
        audio = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            # output_format="mp3_44100_128",
        )

        # Convert generator to bytes
        audio_bytes = b"".join(audio)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            return temp_audio.name
    except Exception as e:
        print(f"Error generating voice response: {e}")
        return None


def initialize_conversation_chain(llm, vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True,
    )


def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone"""
    recorded_audio = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1
    )
    sd.wait()
    return recorded_audio


def transcribe_audio(whisper_model, recorded_audio, sample_rate=44100):
    """Transcribe audio using Whisper"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recorded_audio, sample_rate)
        result = whisper_model.transcribe(temp_audio.name)
        os.unlink(temp_audio.name)
    return result["text"]


def generate_response(chain, query):
    """Generate response using RAG system"""
    if chain is None:
        return "Error: Vector store not initialized"

    response = chain.invoke({"question": query})
    return response["answer"]
