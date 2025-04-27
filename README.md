# DanceCorrect AI - Python Version

DanceCorrect AI is a web application that helps dancers improve their technical dance moves through AI-powered feedback. Using the phone's camera, dancers can record themselves performing a specific move, and the application will analyze their technique and provide detailed corrections.

## Features

- Record dance moves using your phone's camera
- AI-powered analysis of dance technique
- Real-time pose detection and tracking with MediaPipe
- Detailed feedback on posture, timing, and form
- Support for various technical dance moves
- Custom move definitions

## Technology Stack

- Python 3.8+
- Flask web framework
- OpenCV and MediaPipe for pose detection
- OpenAI API for dance analysis
- HTML, CSS (Tailwind via CDN), and JavaScript

## Getting Started

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dance-correct-ai.git
   cd dance-correct-ai/python_app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the python_app directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to use the application.

## How to Use

1. Select a technical dance move from the list or enter a custom move name
2. Position yourself so your full body is visible in the camera frame
3. Click "Record Dance Move" and perform the move when the countdown ends
4. Wait for the AI to analyze your performance
5. Review the detailed feedback and suggestions for improvement
6. Record a new dance move to continue practicing

## Privacy Information

- All video processing is done on your device
- Video recordings are temporarily stored only during analysis
- Pose data is processed locally using MediaPipe
- Data sent to the OpenAI API is only used for generating feedback

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for pose detection
- [OpenAI](https://openai.com/) for LLM-powered analysis
- [Flask](https://flask.palletsprojects.com/) for the web framework 