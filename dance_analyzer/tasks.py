import os
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with minimal configuration
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_video(video_path, feedback_video_path, dance_style):
    """
    Process a dance video and generate feedback.
    
    Args:
        video_path (str): Path to the input video file
        feedback_video_path (str): Path to save the feedback video
        dance_style (str): The dance style being performed
    
    Returns:
        dict: Analysis results including feedback and scores
    """
    try:
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer for feedback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(feedback_video_path, fourcc, fps, (width, height))

        # Process frames
        frames_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark
                frame_data = {
                    'landmarks': [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks],
                    'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
                frames_data.append(frame_data)

                # Draw pose landmarks on frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            out.write(frame)

        cap.release()
        out.release()

        if not frames_data:
            raise Exception("No pose data detected in video")

        # Generate AI analysis
        analysis = get_ai_analysis(frames_data, dance_style)
        
        return {
            'success': True,
            'analysis': analysis,
            'frames_analyzed': len(frames_data)
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def get_ai_analysis(frames_data, dance_style):
    """
    Generate AI analysis of the dance performance.
    
    Args:
        frames_data (list): List of frame data containing pose landmarks
        dance_style (str): The dance style being performed
    
    Returns:
        dict: Analysis results including feedback and scores
    """
    try:
        # Prepare the prompt for OpenAI
        prompt = f"""
        Analyze this dance performance in the {dance_style} style. Consider the following aspects:
        1. Technical execution
        2. Rhythm and timing
        3. Style and expression
        4. Body control and alignment
        5. Energy and performance quality

        Provide specific feedback and suggestions for improvement.
        Rate each aspect on a scale of 1-10.
        """

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional dance instructor providing detailed analysis and feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Extract and format the analysis
        analysis = response.choices[0].message.content

        return {
            'feedback': analysis,
            'raw_data': frames_data  # Include raw data for potential future analysis
        }

    except Exception as e:
        logger.error(f"Error generating AI analysis: {str(e)}")
        return {
            'error': f"Failed to generate analysis: {str(e)}"
        }

def analyze_movement_patterns(pose_frames, dance_style):
    if not pose_frames:
        return {"error": "No pose data available"}

    patterns = {
        "rhythm_consistency": 0,
        "movement_flow": 0,
        "body_control": 0,
        "technique_elements": 0,
        "style_accuracy": 0
    }

    # Analyze based on dance style
    if dance_style == "hiphop":
        patterns.update({
            "body_isolation": 0,
            "groove_quality": 0,
            "power_moves": 0
        })
    elif dance_style == "ballet":
        patterns.update({
            "turnout": 0,
            "pointed_feet": 0,
            "arm_positions": 0
        })
    elif dance_style == "contemporary":
        patterns.update({
            "floor_work": 0,
            "weight_shifts": 0,
            "emotional_expression": 0
        })
    elif dance_style == "jazz":
        patterns.update({
            "sharp_movements": 0,
            "syncopation": 0,
            "style_accuracy": 0
        })
    elif dance_style == "ballroom":
        patterns.update({
            "frame_quality": 0,
            "partner_connection": 0,
            "smooth_transitions": 0
        })
    elif dance_style == "breakdance":
        patterns.update({
            "power_moves": 0,
            "freezes": 0,
            "footwork": 0
        })

    # Analyze movement patterns
    for i in range(1, len(pose_frames)):
        prev = pose_frames[i-1]
        curr = pose_frames[i]
        
        # Calculate movement speed and consistency
        movement_speed = calculate_movement_speed(prev, curr)
        patterns["rhythm_consistency"] += movement_speed
        
        # Calculate body control
        body_stability = calculate_body_stability(curr)
        patterns["body_control"] += body_stability

    # Normalize scores
    for key in patterns:
        patterns[key] = round(patterns[key] / len(pose_frames) * 100, 1)

    return patterns

def calculate_movement_speed(prev_frame, curr_frame):
    # Calculate average movement speed of key points
    total_movement = 0
    key_points = [11, 12, 23, 24]  # Shoulders and hips
    
    for point in key_points:
        prev_pos = prev_frame["landmarks"][point]
        curr_pos = curr_frame["landmarks"][point]
        
        movement = ((curr_pos["x"] - prev_pos["x"])**2 + 
                   (curr_pos["y"] - prev_pos["y"])**2)**0.5
        total_movement += movement
    
    return total_movement / len(key_points)

def calculate_body_stability(frame):
    # Calculate body stability based on key points alignment
    stability = 0
    key_points = [11, 12, 23, 24]  # Shoulders and hips
    
    # Check vertical alignment
    for i in range(len(key_points)-1):
        for j in range(i+1, len(key_points)):
            point1 = frame["landmarks"][key_points[i]]
            point2 = frame["landmarks"][key_points[j]]
            
            # Calculate vertical alignment
            alignment = abs(point1["x"] - point2["x"])
            stability += 1 - alignment
    
    return stability / (len(key_points) * (len(key_points)-1) / 2)

def create_feedback_video(input_path, output_path, pose_frames, feedback):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add timestamp
        cv2.putText(frame, f"Time: {frame_idx/fps:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add feedback text for key moments
        if any(abs(frame_idx/fps - f["timestamp"]) < 0.5 for f in pose_frames):
            cv2.rectangle(frame, (50, height-150), (width-50, height-50),
                        (0, 255, 0), 2)
            
            feedback_lines = feedback.split('.')[:2]
            for i, line in enumerate(feedback_lines):
                y_pos = height - 120 + (i * 30)
                cv2.putText(frame, line.strip(), (60, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release() 