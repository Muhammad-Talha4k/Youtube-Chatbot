import os
import re
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import time

load_dotenv()

# -------------------------
# Configuration & helpers
# -------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Embedding model - supports 50+ languages!
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Text splitting config
COLLECTION_NAME = "youtube_transcripts"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# Retrieval config
TOP_K = 5

# -----------------------------------
# HuggingFace Inference Embeddings
# -----------------------------------

def get_embeddings():
    """Get HuggingFace embeddings using Inference API"""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )

# -----------------------------------
# OpenRouter call helper
# -----------------------------------
from langchain_core.callbacks import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

def call_openrouter(system_prompt: str, user_prompt: str, api_key: str,
                    model: str = None, temperature: float = 0.7, max_retries: int = 3):
    """
    Efficient wrapper for OpenRouter using LangChain's ChatOpenAI with retry logic.
    """
    if not api_key:
        raise ValueError("❌ OPENROUTER_API_KEY not set. Please provide a valid API key.")

    stream_container = st.empty()
    handler = StreamHandler(stream_container)
    
    messages = [("system", system_prompt), ("user", user_prompt)]
    
    for attempt in range(max_retries):
        try:
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                streaming=True,
                callbacks=[handler],
                request_timeout=60,
            )

            # Only invoke ONCE - streaming happens via callback
            llm.invoke(messages)
            
            return handler.text
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit or model overload errors
            if "429" in error_msg or "rate limit" in error_msg or "503" in error_msg or "overloaded" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3  # 3s, 6s, 9s
                    st.warning(f"⚠️ Model is busy (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    handler = StreamHandler(stream_container)  # Reset handler
                    continue
                else:
                    raise Exception("❌ Model is currently overloaded. Please try again in a few moments or switch to a different model.")
            else:
                # Other errors, don't retry
                raise e
    
    raise Exception("Failed to get response after multiple attempts")

# -------------------------
# YouTube Transcript Helpers 
# -------------------------

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any URL pattern."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if not match:
        st.error("❌ Invalid YouTube URL. Could not extract video ID.")
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def get_transcript_any_language(video_id: str):
    """
    Fetch transcript in ANY available language - NO TRANSLATION!
    Multilingual embeddings will handle cross-lingual retrieval.
    """
    yt_api = YouTubeTranscriptApi()

    try:
        # Try English first (most common)
        st.info("🎧 Fetching transcript...")
        transcript_list = yt_api.fetch(video_id, languages=["en"])
        st.success("✅ English transcript found.")
        language = "English"
        
    except NoTranscriptFound:
        try:
            # Get first available transcript in any language
            available = yt_api.list(video_id)
            if not available:
                st.error("❌ No transcripts found for this video.")
                return None, None

            first_transcript = list(available)[0]
            language = first_transcript.language
            lang_code = first_transcript.language_code
            
            st.info(f"🌍 Found transcript in: **{language}** ({lang_code})")
            transcript_list = first_transcript.fetch()
            st.success(f"✅ Transcript loaded in {language}")
            
        except Exception as e:
            st.error(f"❌ Failed to fetch transcript: {e}")
            return None, None

    except TranscriptsDisabled:
        st.error("🚫 Transcripts are disabled for this video.")
        return None, None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None, None

    # Convert to standard format
    transcript = [
        {"text": t.text, "start": t.start, "duration": t.duration}
        for t in transcript_list
    ]
    
    return transcript, language

# -------------------------
# Vector DB helpers
# -------------------------
def ingest_to_qdrant(video_id, youtube_url, transcript, language):
    """Split transcript, embed it, and upload to Qdrant with metadata."""
    if not transcript:
        st.error("❌ No transcript available.")
        return None

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    full_text = " ".join([t["text"] for t in transcript])
    from langchain_core.documents import Document
    docs = [Document(
        page_content=full_text, 
        metadata={
            "video_id": video_id, 
            "source": youtube_url,
            "language": language  # Store original language
        }
    )]

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    # Multilingual embeddings work across all languages!
    embeddings = get_embeddings()

    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    
def query_qdrant(video_id, query_text, top_k=5):
    """Retrieve top-k chunks for a given video_id and query text."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Same embeddings - works across languages!
    embeddings = get_embeddings()

    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k},
    )

    results = retriever.invoke(query_text)
    return results

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="YouTube Transcript Chatbot", layout="wide", page_icon="🎬")

st.markdown(
    "<h1 style='text-align:center;'>🎬 YouTube Transcript Chatbot</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h3 style='text-align: center;'>Chat with YouTube Videos in ANY Language</h3>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    MODEL_OPTIONS = {
        "🦙 Meta LLaMA 3.2-Instruct": "meta-llama/llama-3.2-3b-instruct:free",
        "✨ Mistral Small-Instruct": "mistralai/mistral-small-3.2-24b-instruct:free",
    }

    display_name = st.selectbox(
        "Select Model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="If one model is busy, try switching to another!"
    )

    model_name = MODEL_OPTIONS[display_name]
        
    youtube_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    ingest_btn = st.button("📥 Ingest Transcript", key="ingest_btn", use_container_width=True)

    if ingest_btn:
        if not youtube_url.strip():
            st.warning("Please enter a valid YouTube URL.")
        else:
            with st.spinner("⏳ Fetching transcript..."):
                try:
                    video_id = extract_video_id(youtube_url)
                    transcript, language = get_transcript_any_language(video_id)

                    if transcript:
                        from langchain_core.documents import Document
                        full_text = " ".join([t["text"] for t in transcript])

                        with st.spinner("🔄 Creating embeddings and storing..."):
                            ingest_to_qdrant(video_id, youtube_url, transcript, language)
                        
                        st.session_state["video_id"] = video_id
                        st.session_state["youtube_url"] = youtube_url
                        st.session_state["language"] = language
                        
                        st.success(f"✅ Transcript ingested successfully!")
                        with st.expander("📄 Preview Transcript"):
                            st.code(full_text[:800] + "..." if len(full_text) > 800 else full_text)

                    else:
                        st.error("❌ Failed to fetch transcript.")

                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

# -------------------------
# Chat Section
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "video_id" not in st.session_state:
    st.info("📺 Please ingest a YouTube video first to start chatting.")
    st.info("🌍 **Supports videos in ANY language (Hindi, Urdu, Arabic, Spanish, and 50+ more!)**")
    st.stop()
else:
    if prompt := st.chat_input("Ask about this video in any language..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            docs = query_qdrant(st.session_state.video_id, prompt, top_k=5)
            context = "\n\n".join([d.page_content for d in docs])

            system_prompt = (
                "You are a helpful multilingual assistant that answers questions about YouTube videos. "
                "The transcript context may be in a different language than the question - that's okay! "
                "Read and understand the context in whatever language it's in, then answer the user's question clearly. "
                "If the user asks in English, answer in English. If they ask in another language, answer in that language. "
                "Use only the provided transcript context. If you don't know the answer, say so honestly. "
                "Do NOT mention technical details, metadata, or the language of the transcript."
            )

            user_prompt = f"Transcript Context:\n{context}\n\nQuestion: {prompt}"

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = call_openrouter(
                        system_prompt, 
                        user_prompt, 
                        api_key=os.environ.get("OPENROUTER_API_KEY")
                    )

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error during chat: {e}")
            st.info("💡 Try switching to a different model from the sidebar if you're experiencing errors.")