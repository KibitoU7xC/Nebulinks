from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI()

# Configure Gemini AI (Ensure your API key is correctly set in environment variables)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load trained YOLOv8 model from "models/best.pt"
MODEL_PATH = "models/best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found! Place 'best.pt' inside 'models/' folder.")

model = YOLO(MODEL_PATH)  # Load the trained YOLO model

# Tracker class for player movement
class Tracker:
    def __init__(self, fps=30):
        self.tracked_players = {}

    def update(self, detections):
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if idx not in self.tracked_players:
                self.tracked_players[idx] = {
                    "positions": [(center_x, center_y)],
                    "distance": 0.0,
                    "defending_actions": 0,
                    "shots_taken": 0,
                    "ball_control": 0,
                }
            else:
                prev_x, prev_y = self.tracked_players[idx]["positions"][-1]
                distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                self.tracked_players[idx]["distance"] += distance
                self.tracked_players[idx]["positions"].append((center_x, center_y))

        return self.tracked_players

# Function to process video and track players
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker(fps=cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, classes=0)  # Run YOLOv8 player detection
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0], 'boxes') else []
        tracker.update(boxes)

    cap.release()
    return tracker

# Extract player stats
def extract_player_stats(player_id, tracker):
    if player_id in tracker.tracked_players:
        player_data = tracker.tracked_players[player_id]
        return {
            "Distance": player_data["distance"],
            "Defending": player_data["defending_actions"],
            "Shots": player_data["shots_taken"],
            "Ball Control": player_data["ball_control"],
        }
    return None

# Get analysis from Gemini AI (Updated for Gemini 2.0)
def get_player_analysis(player_id, stats):
    prompt = f"""
    Analyze the performance of Player {player_id} based on:
    - Distance covered: {stats['Distance']} meters
    - Defensive actions: {stats['Defending']}
    - Shots taken: {stats['Shots']}
    - Ball control events: {stats['Ball Control']}

    Provide an expert football performance analysis.
    """

    model = genai.GenerativeModel("gemini-2.0")  # ✅ Updated to Gemini 2.0
    response = model.generate_content(prompt)
    return response.text

# API Endpoint: Upload Video
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_path = f"videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"video_path": video_path}

# API Endpoint: Get Player Stats
@app.post("/player-stats/")
async def get_stats(video_path: str = Form(...), player_id: int = Form(...)):
    tracker = process_video(video_path)
    player_stats = extract_player_stats(player_id, tracker)

    if player_stats:
        analysis = get_player_analysis(player_id, player_stats)
        return {"player_id": player_id, "stats": player_stats, "analysis": analysis}
    return {"error": "Player not found"}
