# YouTube Video Chat Analyzer

An AI-powered application that allows users to have interactive conversations about YouTube videos. The app transcribes video content and uses OpenAI's GPT-4 to answer questions about the video.

## Features

- YouTube video transcription (supports both captions and audio-to-text)
- Interactive chat interface
- Persistent conversation history
- Clean, modern web interface
- Mobile-responsive design

## Setup

1. Clone the repository:
```bash
git clone https://github.com/omarzubair3093/youtube.git
cd youtube
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter your OpenAI API key in the sidebar
2. Paste a YouTube URL
3. Wait for the transcript to load
4. Start asking questions about the video content!

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/).
