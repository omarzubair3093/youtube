import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import whisper
import tempfile
from pydub import AudioSegment
import os
from dotenv import load_dotenv
import warnings
from urllib.parse import urlparse, parse_qs
import yt_dlp  # Added for reliable YouTube downloading

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="▶️",
    layout="wide"
)

# Initialize session state at the start
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

# Custom CSS for chat bubbles
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
        if 'youtu.be' in url:
            video_id = url.split('/')[-1]
        elif 'youtube.com' in url:
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query)['v'][0]
        else:
            st.error("Invalid YouTube URL format")
            return None

        if video_id:
            return video_id.split('&')[0]  # Remove any additional parameters
        return None
    except Exception as e:
        st.error(f"Invalid YouTube URL: {str(e)}")
        return None


def get_video_details(url: str):
    """Get video title and thumbnail."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return None
        # Just return the thumbnail for now to avoid pytube issues
        return {
            'title': f"Video ID: {video_id}",
            'thumbnail': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        }
    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return None


class VideoChat:
    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key."""
        try:
            if not openai_api_key or openai_api_key.isspace():
                raise ValueError("OpenAI API key is required")
            openai.api_key = openai_api_key
            self.whisper_model = None
            self.transcript = ""
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def get_transcript(self, youtube_url: str) -> str:
        """Get transcript using available captions or speech recognition."""
        try:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                return ""

            # First, try to get the transcript from the YouTube Transcript API
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                st.success("Transcript found via YouTube API!")
                return ' '.join([entry['text'] for entry in transcript_list])
            except Exception:
                st.warning("No captions available. Using speech recognition (this may take a while)...")

            # If API fails, fallback to Whisper
            if self.whisper_model is None:
                with st.spinner("Loading Whisper speech recognition model..."):
                    self.whisper_model = whisper.load_model("base")

            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # --- NEW YT-DLP CODE BLOCK ---
                    audio_file_path = os.path.join(temp_dir, "audio")

                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': audio_file_path,
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                        }],
                    }

                    with st.spinner("Downloading audio with yt-dlp..."):
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([youtube_url])

                    downloaded_audio_file = audio_file_path + ".mp3"
                    # --- END OF YT-DLP CODE BLOCK ---

                    # Convert MP3 to WAV for Whisper
                    with st.spinner("Converting audio to WAV format..."):
                        audio = AudioSegment.from_file(downloaded_audio_file)
                        wav_path = os.path.join(temp_dir, "audio.wav")
                        audio.export(wav_path, format="wav")

                    # Transcribe audio using Whisper
                    with st.spinner("Transcribing audio... This may take a few minutes."):
                        result = self.whisper_model.transcribe(wav_path)
                        return result["text"]

                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    return ""

        except Exception as e:
            st.error(f"Error getting transcript: {str(e)}")
            return ""

    def chat_about_video(self, question: str, conversation_history: list) -> str:
        """Chat about the video content."""
        try:
            if not self.transcript:
                return "Please load a video transcript first."

            messages = [
                           {"role": "system",
                            "content": "You are an AI assistant analyzing a video transcript to answer questions. Keep your responses concise and relevant to the video content."},
                           {"role": "system", "content": f"Transcript: {self.transcript}"},
                       ] + conversation_history + [{"role": "user", "content": question}]

            with st.spinner("Thinking..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
            return response.choices[0].message['content']

        except Exception as e:
            st.error(f"Error in chat: {str(e)}")
            return "Sorry, I encountered an error while processing your question. Please try again."


def main():
    st.title("YouTube Video Chat Analyzer")
    st.write("Have an interactive conversation about any YouTube video!")

    # Sidebar for API Key
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "OpenAI API Key:",
            value=os.getenv('OPENAI_API_KEY', ''),
            type="password",
            help="Enter your OpenAI API key or set it in Streamlit Secrets"
        )

        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar!")
            st.stop()

        try:
            if 'video_chat' not in st.session_state:
                st.session_state.video_chat = VideoChat(api_key)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.markdown("---")
        st.markdown("""
        ### 📝 How to use:
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
        youtube_url = st.text_input("🔗 Enter YouTube URL:", value=st.session_state.video_url)

    with col2:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.conversation_history = []
            st.rerun()

    if youtube_url and youtube_url != st.session_state.video_url:
        st.session_state.video_url = youtube_url
        st.session_state.transcript = ""  # Reset transcript for new URL
        st.session_state.conversation_history = []  # Reset history for new video

    if youtube_url:
        # Display video details
        details = get_video_details(youtube_url)
        if details:
            st.image(details['thumbnail'], use_column_width=True)
            st.subheader(details['title'])

        # Get transcript if not already loaded
        if not st.session_state.transcript:
            transcript = st.session_state.video_chat.get_transcript(youtube_url)
            if transcript:
                st.session_state.transcript = transcript
                st.session_state.video_chat.transcript = transcript
                st.success("✅ Transcript loaded! You can now ask questions.")
            else:
                st.error("❌ Failed to load transcript. Please try another video.")
                st.stop()

    # Question input and processing, only if transcript is loaded
    if st.session_state.transcript:
        if question := st.text_input("💬 Ask a question about the video:", key="question_input_widget"):
            # Get response
            response = st.session_state.video_chat.chat_about_video(question, st.session_state.conversation_history)
            if response:
                # Update conversation history
                st.session_state.conversation_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ])

        # Display conversation history
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user-message">
                    <strong>You:</strong>  
{message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong>  
{message["content"]}
                </div>
                ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
