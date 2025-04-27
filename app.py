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
    # Ballet Moves
    "Pirouette",
    "Fouetté",
    "Piqué Turn",
    "Arabesque",
    "Grand Jeté",
    "Attitude",
    "Soutenu",
    "Développé",
    "Chassé",
    "Sauté",
    "Pas de Bourrée",
    "Échappé",
    "Rond de Jambe",
    "Grande Épaulment",
    "Ciseaux",
    "À la seconde turn",

    # Acro Moves
    "Back Handspring",
    "Front Handspring",
    "Aerial",
    "Back Tuck",
    "Front Tuck",
    "Split Leap",
    "Power Jump",
    "Cartwheel",
    "Roundoff",
    "Back Walkover",
    "Front Walkover",
    "Kip-up",
    "Tuck Jump",
    "Butterfly Kick",
    "Needle",

    # Jazz Moves
    "Jazz Pirouette",
    "Chasse",
    "Jazz Split",
    "Piqué Turn",
    "Fan Kick",
    "Jazz Walk",
    "Parallel Turn",
    "Pas de Bourrée",
    "Grand Jeté",
    "Développé",
    "Isolation",
    "Ball Change",
    "Heel Stretch",
    "Pencil Turn",
    "Tendu",

    # Contemporary Moves
    "Floor Work",
    "Isolation",
    "Contract and Release",
    "Plié",
    "Pivot Turn",
    "Choreographic Phrase",
    "Lunge",
    "Leaping Fish",
    "Arabesque",
    "Extension",
    "Parallel Stretch",
    "Roll-Down",
    "Attitude Turn",
    "Side Leap",
    "Crossover",

    # Hip-Hop Moves
    "Popping",
    "Locking",
    "Breaking (B-Boying)",
    "Krumping",
    "Tutting",
    "Waacking",
    "Running Man",
    "Cabbage Patch",
    "Moonwalk",
    "Dougie",
    "Stanky Legg",
    "The Robot",
    "Reebok",
    "Lean Back",
    "Hip-Hop Bounce"
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
        # Process pose data to extract key movement patterns
        movement_patterns = analyze_movement_patterns(pose_data)
        
        # Format the data for the LLM
        formatted_data = {
            'pose_data': pose_data,
            'movement_patterns': movement_patterns,
            'dance_move': dance_move_type
        }
        formatted_data_str = json.dumps(formatted_data, indent=2)
        
        # Prepare the prompt for the LLM
        prompt = f"""
You are a friendly, encouraging dance instructor analyzing a "{dance_move_type}" dance move.

Based on the pose detection data and movement patterns I'm providing, give detailed and specific feedback:

1. ONE positive aspect of their execution (always start positive)
2. TWO specific, easy-to-understand corrections (max 1-2 sentences each)
3. ONE quick tip for immediate improvement
4. A score out of 100 with explanation

Consider these aspects in your analysis:
- Body alignment and posture
- Movement quality and control
- Timing and rhythm
- Spatial awareness
- Technical execution
- Energy and expression

Movement patterns detected:
{json.dumps(movement_patterns, indent=2)}

Use casual, encouraging language that a beginner would understand. Avoid technical jargon.
Keep your total response to 4-5 short sentences maximum.

Here is the detailed data:
{formatted_data_str}
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a supportive dance instructor who gives detailed, encouraging feedback in simple language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error getting AI analysis: {str(e)}")
        return "Sorry, I couldn't analyze your dance move. Please try again."

def detect_dance_move(pose_data):
    try:
        # Process pose data to extract key movement patterns
        movement_patterns = analyze_movement_patterns(pose_data)
        
        # Format the pose data and movement patterns for the LLM
        formatted_data = {
            'pose_data': pose_data,
            'movement_patterns': movement_patterns
        }
        formatted_data_str = json.dumps(formatted_data, indent=2)
        
        # Prepare the prompt for the LLM
        prompt = f"""
You are an expert dance instructor with deep knowledge of ballet, contemporary, and acrobatic dance moves.

Based on the following pose detection data and movement patterns, identify which dance move is being performed from this list:
{DANCE_MOVES}

Consider these aspects:
1. Body positions and angles
2. Movement patterns and timing
3. Weight distribution and balance
4. Spatial patterns and direction
5. Key moments and transitions

Movement patterns detected:
{json.dumps(movement_patterns, indent=2)}

Respond with ONLY the name of the dance move that best matches the data. If you're not confident, respond with "Unknown".
If the move is a variation of a listed move, respond with the closest match from the list.

Here is the detailed pose data:
{formatted_data_str}
"""
        
        # Call OpenAI API with lower temperature for more consistent results
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert dance instructor who can identify dance moves from pose data with high accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=50
        )
        
        detected_move = response.choices[0].message.content.strip()
        return detected_move if detected_move in DANCE_MOVES else "Unknown"
    
    except Exception as e:
        print(f"Error detecting dance move: {str(e)}")
        return "Unknown"

def analyze_movement_patterns(pose_data):
    if not pose_data:
        return {}
    
    # Initialize movement pattern analysis
    patterns = {
        'turns': 0,
        'jumps': 0,
        'leaps': 0,
        'balance_phases': 0,
        'arm_movements': 0,
        'leg_movements': 0,
        'direction_changes': 0
    }
    
    # Key points for analysis
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    
    # Analyze each frame
    for i in range(1, len(pose_data)):
        current_frame = pose_data[i]
        prev_frame = pose_data[i-1]
        
        # Check for turns (rotation around vertical axis)
        if current_frame['landmarks'][LEFT_SHOULDER]['x'] - prev_frame['landmarks'][LEFT_SHOULDER]['x'] > 0.1:
            patterns['turns'] += 1
        
        # Check for jumps (rapid vertical movement)
        if current_frame['landmarks'][LEFT_ANKLE]['y'] - prev_frame['landmarks'][LEFT_ANKLE]['y'] < -0.1:
            patterns['jumps'] += 1
        
        # Check for leaps (one leg extended while other pushes off)
        if (current_frame['landmarks'][LEFT_KNEE]['y'] - prev_frame['landmarks'][LEFT_KNEE]['y'] < -0.1 and
            current_frame['landmarks'][RIGHT_KNEE]['y'] - prev_frame['landmarks'][RIGHT_KNEE]['y'] < -0.1):
            patterns['leaps'] += 1
        
        # Check for balance phases (stable position)
        if (abs(current_frame['landmarks'][LEFT_HIP]['x'] - current_frame['landmarks'][RIGHT_HIP]['x']) < 0.05 and
            abs(current_frame['landmarks'][LEFT_KNEE]['y'] - current_frame['landmarks'][RIGHT_KNEE]['y']) < 0.05):
            patterns['balance_phases'] += 1
        
        # Check for arm movements
        if (abs(current_frame['landmarks'][LEFT_ELBOW]['y'] - prev_frame['landmarks'][LEFT_ELBOW]['y']) > 0.05 or
            abs(current_frame['landmarks'][RIGHT_ELBOW]['y'] - prev_frame['landmarks'][RIGHT_ELBOW]['y']) > 0.05):
            patterns['arm_movements'] += 1
        
        # Check for leg movements
        if (abs(current_frame['landmarks'][LEFT_KNEE]['y'] - prev_frame['landmarks'][LEFT_KNEE]['y']) > 0.05 or
            abs(current_frame['landmarks'][RIGHT_KNEE]['y'] - prev_frame['landmarks'][RIGHT_KNEE]['y']) > 0.05):
            patterns['leg_movements'] += 1
        
        # Check for direction changes
        if (abs(current_frame['landmarks'][LEFT_HIP]['x'] - prev_frame['landmarks'][LEFT_HIP]['x']) > 0.1 or
            abs(current_frame['landmarks'][RIGHT_HIP]['x'] - prev_frame['landmarks'][RIGHT_HIP]['x']) > 0.1):
            patterns['direction_changes'] += 1
    
    return patterns

@app.route('/detect', methods=['POST'])
def detect_and_analyze():
    try:
        # Get the video file
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        # Save the video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        video_file.save(temp_file.name)
        temp_file.close()
        
        # Process the video with MediaPipe Pose
        pose_data = process_video(temp_file.name)
        
        # Detect the dance move
        detected_move = detect_dance_move(pose_data)
        
        # Get AI analysis
        analysis = get_ai_analysis(pose_data, detected_move)
        
        # Clean up
        os.unlink(temp_file.name)
        
        return jsonify({
            'detected_move': detected_move,
            'analysis': analysis
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to analyze dance data'}), 500

if __name__ == '__main__':
    app.run(debug=True) 