# 🎤 Voice Assistant with RAG

A sophisticated voice-powered conversational AI assistant that answers questions based on your uploaded documents using Retrieval-Augmented Generation (RAG). Built with Streamlit, this app combines speech recognition, natural language processing, and text-to-speech capabilities for a seamless voice interaction experience.

## ✨ Features

### 🔧 Core Functionality
- **Document Upload & Processing**: Upload PDF, TXT, and Markdown files to build a custom knowledge base
- **Automatic RAG Setup**: Automatically processes documents, creates vector embeddings, and initializes the conversational AI
- **Voice Input**: Record questions using your microphone with adjustable recording duration
- **Speech-to-Text**: Powered by OpenAI Whisper for accurate transcription
- **Intelligent Responses**: Uses LangChain and OpenAI GPT models with RAG for contextually relevant answers
- **Text-to-Speech Output**: ElevenLabs integration provides natural-sounding voice responses
- **Conversation Memory**: Maintains context throughout the conversation session

### 🎨 User Interface
- **Streamlit Web App**: Clean, responsive web interface accessible from any browser
- **Real-time Status Updates**: Visual feedback during recording, transcription, and response generation
- **Audio Playback**: Built-in audio player for voice responses
- **Conversation History**: View complete chat history with user questions and AI responses
- **Sidebar Configuration**: Easy access to settings and status information

### 🔄 Workflow Automation
- **One-Click Interaction**: Single "Ask a Question" button handles the entire voice pipeline
- **Auto-Initialization**: Automatically processes uploaded documents and sets up the AI assistant
- **Error Handling**: Comprehensive error messages and fallback options
- **Session Management**: Persistent conversation state during the session

### 🛠️ Technical Features
- **Vector Database**: ChromaDB for efficient document storage and retrieval
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Configurable Settings**: Adjustable recording duration and other parameters
- **Cross-Platform**: Works on macOS, Windows, and Linux
- **API Integration**: Leverages OpenAI, ElevenLabs, and Whisper APIs

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Microphone access for voice input
- Speakers/headphones for voice output
- API keys for OpenAI and ElevenLabs

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd ai-engineering-mastery
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

### Running the App

1. **Navigate to the mini-projects directory**:
   ```bash
   cd mini-projects
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run voice_assistant_ui.py
   ```

3. **Open your browser** to the URL shown (typically `http://localhost:8501`)

## 📖 How to Use

### Step 1: Upload Documents
- Use the file uploader in the sidebar to upload PDF, TXT, or Markdown files
- The app will automatically process your documents and initialize the AI assistant
- You'll see confirmation when the assistant is ready

### Step 2: Configure Settings (Optional)
- Adjust the recording duration slider (3-10 seconds) based on your speaking pace
- The default 5 seconds works well for most questions

### Step 3: Ask Questions
- Click the "🎤 Ask a Question" button
- Speak clearly into your microphone
- Wait for the recording to complete (indicated by the progress spinner)
- The app will automatically:
  - Transcribe your speech to text
  - Generate an AI response based on your documents
  - Convert the response to speech

### Step 4: Review Results
- View your transcribed question and the AI's response
- Listen to the voice response if ElevenLabs is configured
- Check the conversation history at the bottom for previous interactions

## 🔑 API Keys Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Add it to your `.env` file as `OPENAI_API_KEY`

### ElevenLabs API Key
1. Visit [ElevenLabs](https://elevenlabs.io/)
2. Create an account
3. Go to Profile → API Keys
4. Generate a new API key
5. Add it to your `.env` file as `ELEVENLABS_API_KEY`

## 🏗️ Architecture

The app consists of two main components:

### `voice_assistant.py` - Backend Logic
- Document loading and preprocessing
- Vector store creation and management
- External API integrations (Whisper, ElevenLabs)
- Conversation chain setup with memory

### `voice_assistant_ui.py` - Frontend Interface
- Streamlit web application
- File upload handling
- Voice recording and playback
- Real-time status updates
- Conversation history display

## 🔧 Configuration Options

### Recording Duration
- **Range**: 3-10 seconds
- **Default**: 5 seconds
- **Purpose**: Controls how long the microphone listens for input

### Supported File Types
- **PDF**: Portable Document Format files
- **TXT**: Plain text files
- **MD**: Markdown files

### Voice Settings
- **Voice ID**: JBFqnCBsd6RMkjVDRZzb (ElevenLabs voice)
- **Model**: eleven_multilingual_v2
- **Language**: English (default)

## 🐛 Troubleshooting

### Common Issues

**"Assistant not initialized"**
- Ensure you've uploaded documents
- Wait for the initialization process to complete
- Check for error messages in the sidebar

**"Could not generate voice response"**
- Verify your ElevenLabs API key is correct
- Check your internet connection
- ElevenLabs service might be temporarily unavailable

**"No documents could be processed"**
- Ensure your files are not corrupted
- Check that files are in supported formats (PDF, TXT, MD)
- Try uploading smaller files first

**Microphone not working**
- Grant microphone permissions to your browser
- Test your microphone in system settings
- Try a different browser or device

### Performance Tips
- Upload smaller documents for faster processing
- Use clear, concise questions for better responses
- Close other applications using the microphone
- Ensure stable internet connection for API calls

## 📚 Dependencies

Key libraries used:
- **Streamlit**: Web app framework
- **LangChain**: LLM orchestration and RAG
- **OpenAI**: GPT models and Whisper
- **ElevenLabs**: Text-to-speech
- **ChromaDB**: Vector database
- **PyPDF2**: PDF processing
- **SoundDevice/SoundFile**: Audio handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the AI Engineering Mastery tutorial series. See the main repository for licensing information.

## 🙏 Acknowledgments

- OpenAI for GPT and Whisper models
- ElevenLabs for text-to-speech technology
- LangChain for RAG framework
- Streamlit for the web app framework
- The open-source community for various libraries used

---

**Built with ❤️ for AI Engineering Mastery**