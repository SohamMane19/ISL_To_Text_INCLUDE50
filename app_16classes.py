import os
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import cv2  # OpenCV for video frame extraction
import tempfile
import mimetypes
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the pre-trained model
model_path = "./best_model.keras"  # Path to the trained .keras file
model = load_model(model_path)

# Define the expected input shape
INPUT_SHAPE = (45, 224, 224, 3)
NBFRAME = 45

# Class labels for the model predictions
CLASSES = [
    "Howareyou", "You", "Thankyou", "GoodMorning", "Extra",
    "Hello", "It", "He", "Loud", "Quiet", "Alright", "Goodafternoon",
    "Happy", "Sad", "Beautiful", "They"
]

# Preprocess the video
def preprocess_video(file_path):
    """
    Preprocess the .mov video file to extract NBFRAME frames, resize them
    to the required input size, and normalize the pixel values.

    Args:
        file_path (str): Path to the uploaded video file.

    Returns:
        np.array: Preprocessed video frames of shape (NBFRAME, 224, 224, 3).
    """
    frames = []
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract frames
    for i in range(NBFRAME):
        frame_pos = int(i * frame_count / NBFRAME)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to model's input shape
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    # Ensure the correct number of frames
    if len(frames) < NBFRAME:
        raise ValueError(f"Insufficient frames in video. Expected {NBFRAME}, got {len(frames)}.")

    # Normalize pixel values
    frames = np.array(frames, dtype="float32") / 255.0
    return np.expand_dims(frames, axis=0)

@app.post("/predict/")
async def predict_sign(file: UploadFile = File(...)):
    """
    Endpoint to upload a .mov file and get the predicted sign.

    Args:
        file (UploadFile): The uploaded .mov video file.

    Returns:
        JSONResponse: The predicted class label.
    """
    # Allowed extensions and MIME types
    allowed_extensions = [".mov"]
    allowed_mime_types = ["video/quicktime"]

    # if file.content_type != "video/quicktime":
    #     raise HTTPException(status_code=400, detail="Invalid file type. Only .mov files are allowed.")

    # Check file extension
    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        return JSONResponse(content={"detail": "Invalid file extension. Only .mov files are allowed."}, status_code=400)

    # Check MIME type
    mimetype, _ = mimetypes.guess_type(file.filename)
    if mimetype not in allowed_mime_types:
        return JSONResponse(content={"detail": f"Invalid MIME type: {mimetype}. Only {allowed_mime_types} are allowed."}, status_code=400)


    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as tmp_file:
        tmp_file.write(await file.read())
        temp_file_path = tmp_file.name

    try:
        # Preprocess the video
        video_data = preprocess_video(temp_file_path)

        # Get predictions from the model
        predictions = model.predict(video_data)
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Return the prediction result
        return JSONResponse({
            "predicted_sign": predicted_class,
            "confidence": float(confidence)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
