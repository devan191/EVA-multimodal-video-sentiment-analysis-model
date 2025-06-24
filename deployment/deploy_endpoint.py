from fastapi import FastAPI, UploadFile, File
from inference import model_fn, predict_fn
import os

app = FastAPI()
model_dict = model_fn(".")

@app.post("/predict/")
async def predict(video: UploadFile = File(...)):
    temp_path = "temp.mp4"
    try:
        with open(temp_path, "wb") as f:
            f.write(await video.read())
        input_data = {"video_path": temp_path}
        result = predict_fn(input_data, model_dict)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)