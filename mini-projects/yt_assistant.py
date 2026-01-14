import os
from unittest import result
import yt_dlp
import whisper
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


load_dotenv()


def embedding_model_selection(model_type="openai"):
    embedding_fn = None
    if model_type == "openai":
        embedding_fn = OpenAIEmbeddings(
            model="text-embedding-3-small",
        )
    elif model_type == "chroma":
        embedding_fn = HuggingFaceEmbeddings()
    elif model_type == "nomic":
        embedding_fn = OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
    else:
        raise ValueError(f"Unsupported embedding type: {model_type}")

    return embedding_fn


def llm_selection(model_type, model_name, temperature=0):
    llm = None
    try:
        if model_type == "openai":
            llm = ChatOpenAI(model=model_name, temperature=temperature)
        elif model_type == "ollama":
            llm = ChatOllama(model=model_name, timeout=120, temperature=temperature)
        else:
            raise ValueError("invalid model type")

        return llm
    except Exception as e:
        print("Model Selection Error: ", e)


def download_video_and_extract_audio(url):
    print("Downloading video...")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "downloads/%(title)s.%(ext)s",
        "http_headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
        video_title = info.get("title", "Unknown Title")
        return audio_path, video_title


def transcribe_audio(audio_path):
    print("Transcribing audio...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def preprocess_audio_text(audio_text, video_title):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # split article text into chunks
    chunks = text_splitter.split_text(audio_text)

    # convert text chunks into langchain documents
    documents = [
        Document(page_content=c, metadata={"source": video_title}) for c in chunks
    ]

    return documents


def create_vector_store(documents, embedding_fn, model_type):
    print(f"Creating vector store using {model_type} embeddings...")

    # Create vector store using LangChain's interface
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        collection_name=f"youtube_summary_{model_type}",
    )


def get_prompts():
    map_prompt_template = """Write a concise summary of the following text:
       "{text}"
       CONCISE SUMMARY:"""

    combine_prompt_template = """Write a detailed summary of the following video transcript sections:
       "{text}"
      
       Include:
       - Main topics and key points
       - Important details and examples
       - Any conclusions or call to action
      
       DETAILED SUMMARY:"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    return map_prompt, combine_prompt


def generate_summary(llm, map_prompt, combine_prompt, documents):
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True,
    )

    summary = chain.invoke(documents)
    return summary


def setup_qa_chain(llm, vector_store):
    """Set up question-answering chain"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True,
    )


def main():
    # use these urls for testing
    urls = [
        "https://www.youtube.com/watch?v=v48gJFQvE1Y&ab_channel=BrockMesarich%7CAIforNonTechies",
        "https://www.youtube.com/watch?v=XwZkNaTYBQI&ab_channel=TheGadgetGameShow%3AWhatTheHeckIsThat%3F%21",
    ]
    # Get model preferences
    print("\nAvailable LLM Models:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama3.2")
    llm_choice = input("Choose LLM model (1/2): ").strip()

    print("\nAvailable Embeddings:")
    print("1. OpenAI")
    print("2. Chroma Default")
    print("3. Nomic (via Ollama)")
    embedding_choice = input("Choose embeddings (1/2/3): ").strip()

    # Configure model settings
    model_type = "openai" if llm_choice == "1" else "ollama"
    model_name = "gpt-4" if llm_choice == "1" else "llama3.2"

    if embedding_choice == "1":
        embedding_type = "openai"
    elif embedding_choice == "2":
        embedding_type = "chroma"
    else:
        embedding_type = "nomic"

    try:
        result = None

        embedding_fn = embedding_model_selection(embedding_type)
        llm = llm_selection(model_type, model_name)

        # Display configuration
        print("\nCurrent Configuration:")
        print(f"LLM: {model_type} ({model_name})")
        print(f"Embeddings: {embedding_type}")

        # Process video
        url = input("\nEnter YouTube URL: ")
        print(f"\nProcessing video...")
        audio_path, video_title = download_video_and_extract_audio(url)
        audio_text = transcribe_audio(audio_path)
        documents = preprocess_audio_text(audio_text, video_title)
        map_prompt, combine_prompt = get_prompts()
        summary = generate_summary(
            llm=llm,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            documents=documents,
        )

        vector_store = create_vector_store(documents, embedding_fn, model_type)
        qa_chain = setup_qa_chain(llm, vector_store)

        os.remove(audio_path)

        result = {
            "summary": summary,
            "title": video_title,
            "full_transcript": audio_text,
        }

        if result:
            print(f"\nVideo Title: {result['title']}")
            print("\nSummary:")
            print(result["summary"])

            # Interactive Q&A
            print("\nYou can now ask questions about the video (type 'quit' to exit)")
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() == "quit":
                    break
                if query:

                    response = qa_chain.invoke({"question": query})
                    print("\nAnswer:", response["answer"])

            # Option to see full transcript
            if input("\nWant to see the full transcript? (y/n): ").lower() == "y":
                print("\nFull Transcript:")
                print(result["full_transcript"])

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure required models and APIs are properly configured.")


if __name__ == "__main__":
    main()
