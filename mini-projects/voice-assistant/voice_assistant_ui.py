import streamlit as st
import os
import time
from voice_assistant import (
    load_documents,
    preprocess_documents,
    create_vector_store,
    initialize_external_clients,
    generate_text_to_speech,
    initialize_conversation_chain,
    record_audio,
    transcribe_audio,
    generate_response,
)
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import tempfile


def process_uploaded_files(uploaded_files):
    """Process uploaded files and return documents"""
    documents = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(temp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            docs = loader.load()
            documents.extend(docs)
            st.sidebar.write(f"Loaded: {uploaded_file.name}")

        except Exception as e:
            st.sidebar.error(f"Error loading {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    return documents


# Initialize session state
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "elevenlabs" not in st.session_state:
    st.session_state.elevenlabs = None
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Page configuration
st.set_page_config(page_title="Voice Assistant", page_icon="🎤", layout="wide")

# Title
st.title("🎤 Voice Assistant")
st.markdown(
    "Upload your documents and chat with an AI assistant using voice input and output. Supports PDF, TXT, and Markdown files."
)

# Sidebar for configuration
st.sidebar.header("Configuration")

# File uploader for documents
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Upload PDF, TXT, or MD files to build the knowledge base",
)

# Show upload status
if uploaded_files:
    st.sidebar.success(f"📁 {len(uploaded_files)} file(s) uploaded")
    for file in uploaded_files:
        st.sidebar.write(f"• {file.name}")
else:
    st.sidebar.info("👆 Upload documents to get started")

# Show initialization status
if st.session_state.initialized:
    st.sidebar.success("🤖 Assistant Ready")
else:
    st.sidebar.info("⏳ Waiting for initialization...")

# Recording duration
recording_duration = st.sidebar.slider(
    "Recording Duration (seconds)",
    min_value=3,
    max_value=10,
    value=5,
    help="How long to record audio for",
)

# Auto-initialize when files are uploaded
if uploaded_files and not st.session_state.initialized:
    with st.spinner("Initializing voice assistant..."):
        try:
            # Process uploaded files
            documents = process_uploaded_files(uploaded_files)
            if not documents:
                st.sidebar.error("No documents could be processed from uploaded files")
            else:
                chunks = preprocess_documents(documents)

                # Create vector store
                persist_dir = "./mini-projects/voice_assistant_db"
                vector_store = create_vector_store(chunks, persist_dir)

                # Initialize LLM and conversation chain
                llm = ChatOpenAI(temperature=0.7)
                conversation_chain = initialize_conversation_chain(llm, vector_store)

                # Initialize external clients
                elevenlabs, whisper_model = initialize_external_clients()

                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.conversation_chain = conversation_chain
                st.session_state.elevenlabs = elevenlabs
                st.session_state.whisper_model = whisper_model
                st.session_state.initialized = True

                st.sidebar.success("Assistant initialized successfully!")

        except Exception as e:
            st.sidebar.error(f"Error initializing assistant: {str(e)}")

# Manual initialize button (fallback)
if st.sidebar.button("Re-initialize Assistant"):
    st.session_state.initialized = False
    st.rerun()

# Main interface
st.subheader("🎤 Voice Assistant")

# Status indicator
if st.session_state.initialized:
    st.success("✅ Assistant is ready! Click the button below to start recording.")
else:
    st.info("👆 Upload documents in the sidebar to get started.")

# Single action button for the entire flow
if st.button("🎤 Ask a Question", type="primary", use_container_width=True):
    if not st.session_state.initialized:
        st.error("Please upload documents and wait for initialization first!")
    else:
        # Step 1: Record
        with st.spinner("🎤 Recording..."):
            recorded_audio = record_audio(duration=recording_duration)

        # Step 2: Transcribe
        with st.spinner("📝 Transcribing..."):
            transcription = transcribe_audio(
                st.session_state.whisper_model, recorded_audio
            )

        # Display transcription
        st.write("**You said:**", transcription)

        # Step 3: Generate response
        with st.spinner("🤖 Thinking..."):
            response = generate_response(
                st.session_state.conversation_chain, transcription
            )

        # Display response
        st.write("**Assistant:**", response)

        # Step 4: Generate and play audio response
        if st.session_state.elevenlabs:
            with st.spinner("🔊 Generating voice response..."):
                audio_file = generate_text_to_speech(
                    st.session_state.elevenlabs, response
                )

            if audio_file:
                st.audio(audio_file, format="audio/mp3")
            else:
                st.warning("Could not generate voice response")
        else:
            st.warning("Voice synthesis not available")

        st.success("Response complete!")

# Chat history
st.markdown("---")
st.subheader("💬 Conversation History")

if st.session_state.conversation_chain and hasattr(
    st.session_state.conversation_chain, "memory"
):
    chat_history = st.session_state.conversation_chain.memory.chat_memory.messages
    if chat_history:
        for i, message in enumerate(chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f"**🗣️ You:** {message.content}")
            else:  # Assistant message
                st.markdown(f"**🤖 Assistant:** {message.content}")
            if i < len(chat_history) - 1:  # Add spacing between exchanges
                st.write("")
    else:
        st.info("No conversation history yet. Click 'Ask a Question' to start!")
else:
    st.info("Assistant not initialized yet.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, OpenAI, Whisper, and ElevenLabs*")
