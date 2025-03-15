from fastapi import FastAPI, UploadFile, File, Form
import requests
import shutil
import os

# Initialize FastAPI app
app = FastAPI()

# Google Colab API URL (Replace this with the ngrok URL from Colab)
COLAB_API_URL = "https://your-colab-url.ngrok.io/process-video/"

# API Endpoint: Upload Video and Process in Colab
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), player_id: int = Form(...)):
    # Save video temporarily
    video_path = f"videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Send video to Google Colab for AI processing
    files = {"file": open(video_path, "rb")}
    data = {"player_id": player_id}
    response = requests.post(COLAB_API_URL, files=files, data=data)

    # Return AI-processed results
    return response.json()
