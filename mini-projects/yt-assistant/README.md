# 🎥 YouTube Video Q&A Assistant

A modern web application that transforms YouTube videos into interactive AI-powered knowledge bases. Ask questions about any YouTube video and get intelligent answers based on the full video transcript.

## ✨ Features

### 🎯 Core Functionality

- **YouTube Video Processing**: Download and transcribe YouTube videos automatically
- **AI-Powered Q&A**: Ask natural language questions about video content
- **Real-time Transcription**: Convert video audio to text using OpenAI Whisper
- **Conversational Interface**: Chat with your videos using modern web UI

### 🤖 AI Model Support

- **Multiple LLMs**:
  - OpenAI GPT-4 (cloud-based)
  - Ollama Llama3.2 (local, privacy-focused)
- **Multiple Embedding Models**:
  - OpenAI text embeddings
  - Chroma default embeddings
  - Nomic embeddings (via Ollama)

### 🎨 User Interface

- **Modern Web Interface**: Built with Streamlit for intuitive interaction
- **Dynamic Model Switching**: Automatically reprocesses videos when you change AI models
- **Chat Interface**: Clean, conversational Q&A experience
- **Sidebar Organization**: Model configuration and transcript access
- **Responsive Design**: Works seamlessly across different screen sizes

### 📊 Advanced Features

- **Vector Database**: ChromaDB for semantic search and retrieval
- **Document Chunking**: Intelligent text splitting for optimal processing
- **Session Management**: Persistent state across interactions
- **Error Handling**: Comprehensive error messages and troubleshooting
- **Automatic Cleanup**: Removes temporary files after processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for OpenAI models)
- Ollama (optional, for local models)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/quamnana/ai-engineering-mastery.git
   cd ai-engineering-mastery
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run mini-projects/yt_assistant_streamlit.py
   ```

## 🎮 Usage

1. **Select AI Models**: Choose your preferred LLM and embedding model from the sidebar
2. **Enter YouTube URL**: Paste any YouTube video URL
3. **Process Video**: Click "Process Video" to transcribe and analyze
4. **Ask Questions**: Use the chat interface to ask questions about the video
5. **View Transcript**: Access the full transcript from the sidebar

## 🏗️ Architecture

### Core Components

- **Video Processing**: yt-dlp for downloading, Whisper for transcription
- **Text Processing**: LangChain for document chunking and embeddings
- **Vector Storage**: ChromaDB for semantic search
- **Q&A Engine**: Retrieval-augmented generation with conversational memory
- **Web Interface**: Streamlit for the user interface

### Data Flow

1. YouTube URL → Audio Download → Transcription
2. Transcript → Document Chunks → Vector Embeddings
3. User Question → Semantic Search → AI Generation → Answer

## 🔧 Configuration

### Model Options

**Language Models:**

- `OpenAI GPT-4`: Most capable, requires API key
- `Ollama Llama3.2`: Local model, privacy-focused

**Embedding Models:**

- `OpenAI`: High quality, requires API key
- `Chroma Default`: Free, uses HuggingFace models
- `Nomic (via Ollama)`: Local embeddings, privacy-focused

### Environment Setup

For Ollama support:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## 📁 Project Structure

```
mini-projects/
├── yt_assistant_streamlit.py    # Main Streamlit application
├── yt_assistant.py              # Command-line version
└── yt_summarizer.py             # Core processing functions
```

## 🐛 Troubleshooting

### Common Issues

**"Model not found" errors:**

- Ensure Ollama is running: `ollama serve`
- Pull required models: `ollama pull llama3.2`

**OpenAI API errors:**

- Check your API key in `.env`
- Verify your OpenAI account has credits

**Video processing fails:**

- Ensure stable internet connection
- Check if the YouTube video is publicly accessible
- Some videos may have restrictions

**Memory issues:**

- Try using smaller embedding models
- Process shorter videos
- Restart the application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **OpenAI** for Whisper and GPT models
- **LangChain** for the RAG framework
- **Streamlit** for the web interface
- **ChromaDB** for vector storage
- **yt-dlp** for YouTube downloading

---

**Built with ❤️ for AI Engineering Mastery**
