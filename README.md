# 🎬 YouTube Transcript Chatbot

> An intelligent multilingual chatbot that enables natural conversations with YouTube videos in 50+ languages using RAG (Retrieval Augmented Generation).

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com/)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🌟 Overview

YouTube Transcript Chatbot is an AI-powered application that allows users to have natural language conversations with YouTube videos. The system extracts video transcripts, processes them using advanced NLP techniques, and enables semantic search and question-answering in multiple languages.

## 📁 Project Structure

```
youtube-chatbot/
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── docker/
│   ├── Dockerfile             # Docker container configuration
│   └── .dockerignore          # Docker ignore rules
└── README.md                  # Readme file
```

### Key Highlights

- **🌍 Multilingual Support**: Works seamlessly with videos in English, Hindi, Urdu, Arabic, Spanish, and 50+ other languages
- **🚀 No Translation Required**: Uses original transcripts for maximum accuracy and preserves cultural context
- **💬 Natural Conversations**: Powered by state-of-the-art LLMs from OpenRouter
- **🎯 Smart Retrieval**: Employs RAG with vector embeddings for precise context retrieval
- **⚡ Fast & Efficient**: Cloud-based embeddings with no local model downloads

## ✨ Features

### Core Functionality
- 📺 **YouTube Integration**: Automatic transcript extraction from any YouTube video
- 🔍 **Semantic Search**: Intelligent context retrieval using vector similarity
- 💡 **Conversational AI**: Natural dialogue with multiple LLM options
- 🌐 **Cross-lingual Queries**: Ask questions in any language, get relevant answers

### Technical Features
- 🎯 **RAG Architecture**: Retrieval Augmented Generation for accurate responses
- 🔄 **Streaming Responses**: Real-time token streaming for better UX
- 🔁 **Automatic Retry Logic**: Handles API rate limits and timeouts gracefully
- 📊 **Vector Database**: Persistent storage with Qdrant Cloud
- 🧠 **Multiple LLM Support**: Choose from Mistral & LLaMA

## 🎥 Demo

<!-- Add screenshots or GIF here -->

```bash
# Quick start
streamlit run app.py
```

## 🏗️ Architecture

```
┌─────────────────────┐
│  Youtube Video URL  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Transcript Extractor│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Text Splitter     │ (600 char chunks)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HuggingFace API    │ (Multilingual Embeddings)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Qdrant Vector DB  │
└────────┬────────────┘
         │
         ▼
    User Query
         │
         ▼
┌─────────────────────┐
│  Semantic Search    │ (MMR Retrieval)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  OpenRouter LLM     │ (Generate Response)
└────────┬────────────┘
         │
         ▼
    Streamed Answer
```

## 🛠️ Tech Stack

### Frontend
- **[Streamlit](https://streamlit.io/)** - Interactive web application framework

### Backend & AI
- **[LangChain](https://langchain.com/)** - LLM orchestration framework
- **[OpenRouter](https://openrouter.ai/)** - Unified LLM API gateway
- **[HuggingFace](https://huggingface.co/)** - Multilingual embeddings API
- **[Qdrant](https://qdrant.tech/)** - Vector database for semantic search

### Core Libraries
- **youtube-transcript-api** - YouTube transcript extraction
- **sentence-transformers** - Multilingual embeddings model
- **langchain-openai** - OpenRouter integration
- **langchain-qdrant** - Qdrant vector store integration

## 📦 Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- API keys (see [API Keys Setup](#-api-keys-setup))

## 🚀 Installation

### Method 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-chatbot.git
cd youtube-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run app.py
```

### Method 2: Docker

```bash
# Build the image
docker build -t youtube-chatbot .

# Run the container
docker run -d \
  --name yt-chatbot \
  -p 8501:8501 \
  --env-file .env \
  youtube-chatbot
```

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
# OpenRouter API Key (for LLM generation)
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# Qdrant Cloud Configuration
QDRANT_URL=https://xxxxx.qdrant.io
QDRANT_API_KEY=xxxxx

# HuggingFace API Key (for embeddings)
HUGGINGFACE_API_KEY=hf_xxxxx
```

### Configuration Options

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | API key for LLM access | ✅ Yes |
| `QDRANT_URL` | Qdrant cluster URL | ✅ Yes |
| `QDRANT_API_KEY` | Qdrant authentication key | ✅ Yes |
| `HUGGINGFACE_API_KEY` | HuggingFace API token | ✅ Yes |

## 📖 Usage

### Basic Workflow

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Ingest a Video**
   - Paste a YouTube URL in the sidebar
   - Click "📥 Ingest Transcript"
   - Wait for processing (10-30 seconds)

3. **Start Chatting**
   - Type your question in the chat input
   - Receive AI-generated answers based on video content
   - Continue the conversation naturally

### Supported Video Languages

The chatbot supports transcripts in:
- 🇬🇧 English
- 🇮🇳 Hindi
- 🇵🇰 Urdu
- 🇸🇦 Arabic
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇯🇵 Japanese
- 🇰🇷 Korean
- 🇨🇳 Chinese
- And 50+ more languages!

### Available LLM Models

Choose from multiple AI models:

| Model | Provider | Strengths |
|-------|----------|-----------|
| LLaMA 3.2 3B | Meta | Fast, efficient |
| Mistral Small | Mistral AI | Balanced performance |


## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs**: Open an issue describing the problem
- 💡 **Suggest Features**: Share your ideas for improvements
- 📝 **Improve Documentation**: Help make docs clearer
- 🔧 **Submit Pull Requests**: Fix bugs or add features

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/youtube-chatbot.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "Add amazing feature"

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Technologies
- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [LangChain](https://langchain.com/) - For LLM orchestration
- [OpenRouter](https://openrouter.ai/) - For unified LLM access
- [Qdrant](https://qdrant.tech/) - For vector database
- [HuggingFace](https://huggingface.co/) - For embeddings API

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Talha4k/youtube-chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Talha4k/youtube-chatbot/discussions)
- **Email**: muhammadtalhasheikh50@gmail.com


<div align="center">

**Built with ❤️ by [Muhammad Talha](https://github.com/Talha4k)**

⭐ Star this repo if you find it helpful!

</div>
