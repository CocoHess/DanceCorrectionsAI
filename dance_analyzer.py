import os
import cv2
import json
import mediapipe as mp
import openai
import streamlit as st
import tempfile
import numpy as np

from datetime import datetime

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_video(video_path, feedback_video_path, dance_style):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = max(int(fps * 0.5), 1)  # Sample every 0.5 seconds
    max_frames = 15
    pose_frames = []
    frame_idx = 0

    while len(pose_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frame = cv2.resize(frame, (160, 120))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                landmarks = [
                    {
                        "x": round(l.x, 2),
                        "y": round(l.y, 2),
                        "z": round(l.z, 2),
                        "visibility": round(l.visibility, 2)
                    }
                    for l in results.pose_landmarks.landmark
                ]
                pose_frames.append({
                    "timestamp": round(frame_idx / fps, 2),
                    "landmarks": landmarks
                })
        frame_idx += 1
    cap.release()

    if not pose_frames:
        return "No pose data detected.", []

    # Analyze movement patterns
    movement_patterns = analyze_movement_patterns(pose_frames, dance_style)

    # Generate feedback via OpenAI
    prompt = f"""Analyze this {dance_style} dance performance and provide feedback:

Movement Analysis:
{json.dumps(movement_patterns, indent=2)}

Please provide:
1. ONE positive aspect of the performance
2. TWO specific corrections for {dance_style} technique
3. ONE quick tip to improve
4. A score out of 100

Keep the response concise and focused on {dance_style} dance style."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a supportive dance instructor specializing in multiple dance styles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    feedback = response.choices[0].message.content

    # Create feedback video with annotations
    create_feedback_video(video_path, feedback_video_path, pose_frames, feedback)

    return feedback, movement_patterns

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

    style_keys = {
        "hiphop": ["body_isolation", "groove_quality", "power_moves"],
        "ballet": ["turnout", "pointed_feet", "arm_positions"],
        "contemporary": ["floor_work", "weight_shifts", "emotional_expression"],
        "jazz": ["sharp_movements", "syncopation", "style_accuracy"],
        "ballroom": ["frame_quality", "partner_connection", "smooth_transitions"],
        "breakdance": ["power_moves", "freezes", "footwork"]
    }

    for key in style_keys.get(dance_style, []):
        patterns[key] = 0

    for i in range(1, len(pose_frames)):
        prev = pose_frames[i - 1]
        curr = pose_frames[i]

        speed = calculate_movement_speed(prev, curr)
        control = calculate_body_stability(curr)

            # Normalize scores
    for key in patterns:
        patterns[key] = round(patterns[key] / len(pose_frames) * 100, 1)

    return patterns

