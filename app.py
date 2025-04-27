import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import base64
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DANCE_MOVES = [
    'Axel Turn',
    'Ball Change',
    'Barrel Turn',
    'Chassé',
    'Fan Kick',
    'Hitch Kick',
    'Lindy Hop',
    'Passé',
    'Pirouette',
    'Plié',
    'Relevé',
    'Grand Jeté',
    'Fouetté'
]

@app.route('/')
def index():
    return render_template('index.html', dance_moves=DANCE_MOVES)

@app.route('/analyze', methods=['POST'])
def analyze_dance():
    try:
        # Get the video file and dance move type
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        dance_move_type = request.form.get('dance_move_type', '')
        
        if not dance_move_type:
            return jsonify({'error': 'No dance move type provided'}), 400
        
        # Save the video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        video_file.save(temp_file.name)
        temp_file.close()
        
        # Process the video with MediaPipe Pose
        pose_data = process_video(temp_file.name)
        
        # Get AI analysis
        analysis = get_ai_analysis(pose_data, dance_move_type)
        
        # Clean up
        os.unlink(temp_file.name)
        
        return jsonify({'analysis': analysis})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to analyze dance data'}), 500

def process_video(video_path):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # We'll capture pose data every few frames to reduce computational load
        sample_interval = max(int(fps / 5), 1)  # 5 samples per second
        
        pose_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract pose landmarks
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    # Add frame timestamp and landmarks to our collection
                    timestamp = frame_idx / fps
                    pose_frames.append({
                        'timestamp': timestamp,
                        'landmarks': landmarks
                    })
            
            frame_idx += 1
        
        cap.release()
        return pose_frames
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return []

def get_ai_analysis(pose_data, dance_move_type):
    try:
        # Format the pose data for the LLM
        formatted_pose_data = json.dumps(pose_data, indent=2)
        
        # Prepare the prompt for the LLM
        prompt = f"""
You are a friendly, encouraging dance instructor analyzing a "{dance_move_type}" dance move.

Based on the pose detection data I'm providing, give BRIEF and SIMPLE feedback:

1. ONE positive aspect of their execution (always start positive)
2. TWO specific, easy-to-understand corrections (max 1-2 sentences each)
3. ONE quick tip for immediate improvement

Use casual, encouraging language that a beginner would understand. Avoid technical jargon.
Keep your total response to 4-5 short sentences maximum.

Here is the pose data:
{formatted_pose_data}
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a supportive dance instructor who gives concise, encouraging feedback in simple language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error getting AI analysis: {str(e)}")
        return "Sorry, I couldn't analyze your dance move. Please try again."

if __name__ == '__main__':
    app.run(debug=True) 