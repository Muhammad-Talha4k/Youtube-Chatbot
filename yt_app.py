# streamlit_app.py
import os
import re
import math 
import streamlit as st
# LangChain + loaders + utils
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_openai import ChatOpenAI
from deep_translator import GoogleTranslator
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Configuration & helpers
# -------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# OpenRouter endpoint & model - you can change model to whatever is available on your OpenRouter plan
OPENROUTER_MODELS = ["mistralai/mistral-small-3.2-24b-instruct:free",
                    "meta-llama/llama-3.2-3b-instruct:free"]  

# Embedding model (multilingual Minilm)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Text splitting config
COLLECTION_NAME = "youtube_transcripts"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Retrieval config
TOP_K = 5

# -----------------------------------
# OpenRouter call helper
# -----------------------------------
from langchain_core.callbacks import BaseCallbackHandler

# ---- Streamlit callback for live updates ----
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
import time
def call_openrouter(system_prompt: str, user_prompt: str, api_key: str,
                    model: str = None, temperature: float = 0.7):
    """
    Efficient wrapper for OpenRouter using LangChain's ChatOpenAI.
    Uses the model selected in the sidebar unless overridden.
    """
    if not api_key:
        raise ValueError("❌ OPENROUTER_API_KEY not set. Please provide a valid API key.")

    # ✅ Use sidebar model if not explicitly passed
    #active_model = model or model_name

    # ✅ Stream container in UI
    stream_container = st.empty()
    handler = StreamHandler(stream_container)
    
    # ✅ Initialize LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,  # fixed temperature (not user-controlled)
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        streaming=True,
        callbacks=[handler],  # 👈 important
        request_timeout=60,  # <— add this line                
    )

    # ✅ Combine system + user messages
    messages = [("system", system_prompt), ("user", user_prompt)]

    # ✅ Generate completion
    response = llm.invoke(messages)

    # ✅ Just invoke to trigger streaming, but don’t print returned content again
    llm.invoke(messages)

    # ✅ Only return the streamed text
    return handler.text

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

# ... (Your extract_video_id function here) ...

def get_transcript_in_english(video_id: str):
    """
    Fetch transcript for given video_id.
    If English not found, auto-translates to English using batch translation 
    and shows a progress bar.
    """
    ytt_api = YouTubeTranscriptApi()

    try:
        st.info("🎧 Trying to fetch English transcript...")
        # Try to fetch English directly
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        st.success("✅ English transcript found.")
        # Return in the desired format
        return [
            {"text": t.text, "start": t.start, "duration": t.duration}
            for t in transcript_list
        ]

    except NoTranscriptFound:
        try:
            st.warning("⚠️ English transcript not found. Looking for other languages...")
            available = ytt_api.list(video_id)
            if not available:
                st.error("❌ No transcripts found for this video.")
                return None

            first_transcript = list(available)[0]
            lang = first_transcript.language_code
            st.warning(f"Found transcript in '{lang}'. Fetching...")
            
            # This 'transcript' is a list of FetchedTranscriptSnippet objects
            transcript = first_transcript.fetch() 
            total_lines = len(transcript)

            st.info(f"🌐 Translating {total_lines} lines from '{lang}' to English...")
            
            # --- START OF NEW PROGRESS/BATCHING LOGIC ---
            
            # 1. Initialize progress bar and final list
            progress_text = "Translation starting..."
            progress_bar = st.progress(0, text=progress_text)
            translated_transcript = []
            
            # 2. Set chunk size and translator
            translator = GoogleTranslator(source="auto", target="en")
            chunk_size = 100  # Translate 50 lines at a time
            num_chunks = math.ceil(total_lines / chunk_size)

            # 3. Loop over the *original* transcript list in chunks
            for i in range(0, total_lines, chunk_size):
                # Calculate progress
                percent_complete = (i / total_lines)
                progress_text = f"Translating chunk {i//chunk_size + 1} of {num_chunks}... ({int(percent_complete * 100)}%)"
                progress_bar.progress(percent_complete, text=progress_text)

                # Get the chunk of *objects*
                transcript_chunk = transcript[i : i + chunk_size]
                
                # Get the text *from* that chunk
                texts_to_translate = [t.text for t in transcript_chunk if hasattr(t, 'text')]

                if not texts_to_translate:
                    # This chunk had no text, just skip it
                    continue
                
                # Translate this chunk
                translated_texts = translator.translate_batch(texts_to_translate)
                
                # Rebuild this chunk with original timings
                text_index = 0
                for t in transcript_chunk:
                    if hasattr(t, 'text'):
                        translated_transcript.append({
                            "text": translated_texts[text_index],
                            "start": t.start,
                            "duration": t.duration,
                        })
                        text_index += 1
            
            # 4. Clean up
            progress_bar.progress(1.0, text="Translation complete!")
            progress_bar.empty() # Remove the progress bar after completion
            
            st.success("✅ Transcript translated to English.")
            return translated_transcript

        except Exception as e:
            st.error(f"❌ Failed to fetch or translate transcript: {e}")
            if 'progress_bar' in locals(): # Remove progress bar on error
                progress_bar.empty()
            return None

    except TranscriptsDisabled:
        st.error("🚫 Transcripts are disabled for this video.")
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

def transcript_to_documents(transcript, video_id):
    """Convert transcript data into LangChain Document objects."""
    if not transcript:
        st.error("❌ No transcript available.")
        return None

    full_text = " ".join([t["text"] for t in transcript if t.get("text")])
    return [Document(page_content=full_text, metadata={"video_id": video_id})]

# -------------------------
# Vector DB helpers
# -------------------------
def ingest_to_qdrant(video_id, youtube_url, transcript):
    """Split transcript, embed it, and upload to Qdrant with metadata."""
    if not transcript:
        st.error("❌ No transcript available.")
        return None

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Prepare data
    full_text = " ".join([t["text"] for t in transcript])
    docs = [Document(page_content=full_text, metadata={"video_id": video_id, "source": youtube_url})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. CREATE COLLECTION & INGEST DATA
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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

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
    "<h3 style='text-align: center;'>Chat with your YouTube Video's</h3>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")

    # ✅ Define friendly display names mapped to full OpenRouter IDs
    MODEL_OPTIONS = {
        " Meta LLaMA 3.2-Instruct": "meta-llama/llama-3.2-3b-instruct:free",
        "Mistral Small-Instruct": "mistralai/mistral-small-3.2-24b-instruct:free",
    }

    # ✅ Show friendly names in dropdown
    display_name = st.selectbox(
        "Select Model",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )

    # ✅ Use full model ID internally
    model_name = MODEL_OPTIONS[display_name]

    youtube_url = st.text_input("Enter YouTube Video URL",placeholder="https://www.youtube.com/watch?v=...")
    ingest_btn = st.button("Ingest Transcript", key="ingest_btn")

    if ingest_btn:
        if not youtube_url.strip():
            st.warning("Please enter a valid YouTube URL.")
        else:
            # 👇 Entire process now lives inside sidebar spinner
            with st.spinner("⏳ Fetching and preparing transcript..."):
                try:
                    video_id = extract_video_id(youtube_url)

                    # Run transcript + translation fully in sidebar
                    transcript = get_transcript_in_english(video_id)

                    if transcript:
                        from langchain_core.documents import Document
                        full_text = " ".join([t["text"] for t in transcript])
                        docs = [Document(page_content=full_text, metadata={"video_id": video_id, "source": youtube_url})]

                        ingest_to_qdrant(video_id, youtube_url, transcript)
                        st.session_state["video_id"] = video_id  # ✅ store for chat
                        st.session_state["youtube_url"] = youtube_url
                        st.success("✅ Transcript ingested and stored successfully.")
                        st.write("🎬 Video ID:", video_id)
                        st.code(full_text[:600] + " ...")

                    else:
                        st.error("❌ Failed to fetch or translate transcript.")

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
    st.stop()
else:
    if prompt := st.chat_input("Ask about this video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            docs = query_qdrant(st.session_state.video_id, prompt, top_k=5)
            # 💡 DEBUG STEP: Display the retrieved documents
            st.sidebar.subheader("Debugging: Retrieved Chunks")
            for i, d in enumerate(docs):
                st.sidebar.write(f"Chunk {i+1} Score: {d.metadata.get('score', 'N/A')}")
                st.sidebar.code(d.page_content[:250] + "...")
            context = "\n\n".join([d.page_content for d in docs])

            system_prompt = (
            "You are a helpful assistant that answers questions about YouTube videos. "
            "Use only the provided transcript context to answer clearly and naturally. "
            "Do NOT mention technical details or metadata. "
            "If you don't know the answer, say you don't know and don't hallucinate."
            )

            user_prompt = f"Transcript Context:\n{context}\n\nQuestion: {prompt}"

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = call_openrouter(system_prompt, 
                                             user_prompt, 
                                             api_key=os.environ.get("OPENROUTER_API_KEY"))
                    st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error during chat: {e}")