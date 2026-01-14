import streamlit as st
import os
from yt_summarizer import (
    embedding_model_selection,
    llm_selection,
    download_video_and_extract_audio,
    transcribe_audio,
    preprocess_audio_text,
    create_vector_store,
    get_prompts,
    generate_summary,
    setup_qa_chain,
)

st.set_page_config(page_title="YouTube Video Q&A Assistant", page_icon="🎥")

st.title("🎥 YouTube Video Q&A Assistant")

# Initialize session state FIRST
if "processed" not in st.session_state:
    st.session_state.processed = False
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "title" not in st.session_state:
    st.session_state.title = ""
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_config" not in st.session_state:
    st.session_state.last_config = {}
if "last_url" not in st.session_state:
    st.session_state.last_url = ""
if "show_transcript" not in st.session_state:
    st.session_state.show_transcript = False

# Sidebar for model configuration
st.sidebar.header("Model Configuration")

# LLM Selection
st.sidebar.subheader("Language Model")
llm_choice = st.sidebar.selectbox(
    "Choose LLM:", ["OpenAI GPT-4", "Ollama Llama3.2"], index=0
)

# Embedding Selection
st.sidebar.subheader("Embeddings")
embedding_choice = st.sidebar.selectbox(
    "Choose Embeddings:", ["OpenAI", "Chroma Default", "Nomic (via Ollama)"], index=0
)

# Map choices to model types
model_type = "openai" if llm_choice == "OpenAI GPT-4" else "ollama"
model_name = "gpt-4" if llm_choice == "OpenAI GPT-4" else "llama3.2"

if embedding_choice == "OpenAI":
    embedding_type = "openai"
elif embedding_choice == "Chroma Default":
    embedding_type = "chroma"
else:
    embedding_type = "nomic"

# Store current model configuration
current_config = {"llm_choice": llm_choice, "embedding_choice": embedding_choice}

# Check if model configuration changed
if st.session_state.last_config != current_config:
    # Model configuration changed, reset processed state
    st.session_state.processed = False
    st.session_state.summary = ""
    st.session_state.title = ""
    st.session_state.transcript = ""
    st.session_state.qa_chain = None
    st.session_state.messages = []  # Also reset chat history
    st.session_state.last_config = current_config
    if (
        st.session_state.last_config != {}
    ):  # Only show message if this isn't the first run
        st.info(
            "Model configuration changed. Please reprocess the video with the new settings."
        )

# Display current configuration
st.sidebar.subheader("Current Configuration")
st.sidebar.write(f"**LLM:** {llm_choice}")
st.sidebar.write(f"**Embeddings:** {embedding_choice}")

# Full Transcript Section (in sidebar)
if st.session_state.processed:
    with st.sidebar:
        st.subheader("📄 Full Transcript")
        if st.button("Show/Hide Full Transcript"):
            if "show_transcript" not in st.session_state:
                st.session_state.show_transcript = False
            st.session_state.show_transcript = not st.session_state.show_transcript

        if st.session_state.get("show_transcript", False):
            st.text_area("Transcript", st.session_state.transcript, height=300)

# Main content
st.header("Enter YouTube URL")
url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

# Check if URL changed
if "last_url" not in st.session_state:
    st.session_state.last_url = url
elif st.session_state.last_url != url and url != "":
    # URL changed, reset processed state
    st.session_state.processed = False
    st.session_state.summary = ""
    st.session_state.title = ""
    st.session_state.transcript = ""
    st.session_state.qa_chain = None
    st.session_state.messages = []
    st.session_state.last_url = url
    st.info("URL changed. Please reprocess the video.")

# Process button
if st.button("Process Video", type="primary"):
    if url:
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                # Configure models
                embedding_fn = embedding_model_selection(embedding_type)
                llm = llm_selection(model_type, model_name)

                # Process video
                audio_path, video_title = download_video_and_extract_audio(url)
                audio_text = transcribe_audio(audio_path)
                documents = preprocess_audio_text(audio_text, video_title)

                # Generate summary
                map_prompt, combine_prompt = get_prompts()
                summary = generate_summary(llm, map_prompt, combine_prompt, documents)

                # Create vector store and QA chain
                vector_store = create_vector_store(documents, embedding_fn, model_type)
                qa_chain = setup_qa_chain(llm, vector_store)

                # Clean up audio file
                os.remove(audio_path)

                # Store results in session state
                st.session_state.processed = True
                st.session_state.summary = summary
                st.session_state.title = video_title
                st.session_state.transcript = audio_text
                st.session_state.qa_chain = qa_chain

                st.success("Video processed successfully!")

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.info("Make sure required models and APIs are properly configured.")
    else:
        st.warning("Please enter a YouTube URL.")

# Display results if processed
if st.session_state.processed:
    st.header(f"📹 {st.session_state.title}")

    # st.subheader("📝 Summary")
    # st.write(st.session_state.summary)

    # Q&A Section
    st.header("❓ Ask Questions About the Video")

    # Chat interface
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"question": prompt})
                    answer = response["answer"]
                    st.markdown(answer)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, and Whisper*")
