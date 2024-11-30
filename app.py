import streamlit as st
import pytube
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import whisper
import tempfile
from pydub import AudioSegment
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f2f6;
    }
    .video-preview {
        width: 100%;
        max-width: 600px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    try:
        return pytube.YouTube(url).video_id
    except Exception as e:
        st.error(f"Invalid YouTube URL: {str(e)}")
        return None

def get_video_details(url: str):
    """Get video title and thumbnail."""
    try:
        yt = pytube.YouTube(url)
        return {
            'title': yt.title,
            'thumbnail': yt.thumbnail_url
        }
    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return None

class VideoChat:
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=openai_api_key)
        self.whisper_model = None
        self.transcript = ""
        
    @st.cache_data
    def get_transcript(self, youtube_url: str) -> str:
        """Get transcript using available captions or speech recognition."""
        try:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                return ""
                
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                return ' '.join([entry['text'] for entry in transcript_list])
            except Exception as e:
                st.warning("No captions available. Using speech recognition (this may take a while)...")
                
                if self.whisper_model is None:
                    with st.spinner("Loading Whisper model..."):
                        self.whisper_model = whisper.load_model("base")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download audio
                    yt = pytube.YouTube(youtube_url)
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    audio_file = audio_stream.download(output_path=temp_dir)
                    
                    # Convert to WAV
                    audio = AudioSegment.from_file(audio_file)
                    wav_path = os.path.join(temp_dir, "audio.wav")
                    audio.export(wav_path, format="wav")
                    
                    # Transcribe
                    with st.spinner("Transcribing audio..."):
                        result = self.whisper_model.transcribe(wav_path)
                        return result["text"]
                    
        except Exception as e:
            st.error(f"Error getting transcript: {str(e)}")
            return ""

    def chat_about_video(self, question: str, conversation_history: list) -> str:
        """Chat about the video content."""
        try:
            messages = [
                {"role": "system", "content": "You are analyzing a video transcript to answer questions. Keep your responses concise and relevant to the video content."},
                {"role": "system", "content": f"Transcript: {self.transcript}"},
            ] + conversation_history + [{"role": "user", "content": question}]

            with st.spinner("Thinking..."):
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
            
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"Error in chat: {str(e)}")
            return ""

def main():
    st.title("💭 YouTube Video Chat Analyzer")
    st.write("Have an interactive conversation about any YouTube video!")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'video_url' not in st.session_state:
        st.session_state.video_url = ""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "OpenAI API Key:", 
            value=os.getenv('OPENAI_API_KEY', ''),
            type="password",
            help="Enter your OpenAI API key. It will not be stored."
        )
        
        if api_key:
            st.session_state.video_chat = VideoChat(api_key)
        
        st.markdown("---")
        st.markdown("""
        ### 📖 How to use:
        1. Enter your OpenAI API key
        2. Paste a YouTube URL
        3. Wait for transcript to load
        4. Start asking questions!
        
        ### 🤔 Example questions:
        - "What are the main topics discussed?"
        - "Can you summarize the key points?"
        - "What was the conclusion?"
        """)

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        youtube_url = st.text_input("🎥 Enter YouTube URL:", value=st.session_state.video_url)
    
    with col2:
        if st.button("🔄 Clear Chat", type="secondary"):
            st.session_state.conversation_history = []
            st.rerun()
    
    if youtube_url and youtube_url != st.session_state.video_url:
        st.session_state.video_url = youtube_url
        if 'transcript' in st.session_state:
            del st.session_state.transcript
    
    if youtube_url and 'video_chat' in st.session_state:
        # Display video details
        details = get_video_details(youtube_url)
        if details:
            st.image(details['thumbnail'], use_column_width=True)
            st.subheader(details['title'])
        
        # Get transcript if not already loaded
        if 'transcript' not in st.session_state:
            with st.spinner("📝 Loading transcript..."):
                st.session_state.transcript = st.session_state.video_chat.get_transcript(youtube_url)
                st.session_state.video_chat.transcript = st.session_state.transcript
            if st.session_state.transcript:
                st.success("✅ Transcript loaded! You can now ask questions.")
        
        # Question input
        question = st.text_input("💬 Ask a question about the video:", key="question_input")
        
        if question:
            # Get response
            response = st.session_state.video_chat.chat_about_video(
                question, 
                st.session_state.conversation_history
            )
            
            # Update conversation history
            st.session_state.conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ])
            
            # Clear question input
            st.session_state.question_input = ""

        # Display conversation history
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
