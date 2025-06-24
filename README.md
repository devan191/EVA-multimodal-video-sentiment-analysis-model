# Video Sentiment Analysis API

This project provides a multimodal sentiment and emotion analysis API for video files. It uses deep learning models to analyze video, audio, and transcribed text from `.mp4` files, returning the top predicted emotions and sentiments for each utterance in the video.

## Features

- Accepts `.mp4` video uploads via a REST API
- Extracts video frames, audio features, and transcribes speech
- Predicts emotions and sentiments for each utterance
- Built with FastAPI for easy deployment and testing
- Ready for deployment on Render, or local use

## Project Structure

```
video-sentiment-model/
├── deployment/
│   ├── deploy_endpoint.py   # FastAPI app (main API server)
│   ├── inference.py         # Model loading and prediction logic
│   ├── requirements.txt     # Python dependencies
│   └── render.yaml          # Render deployment config
├── training/                # Model training scripts (optional for inference)
├── dataset/                 # (Ignored) Training/validation/test data
├── .gitignore               # Prevents large files from being tracked
└── README.md                # This file
```

## Setup & Local Testing

1. **Clone the repository:**
   ```
   git clone <your-repo-url>
   cd video-sentiment-model/deployment
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Ensure model weights are present:**
   - Place your trained model checkpoint (e.g., `checkpoint.pth`) in `deployment/saved_models/`.
   - The API expects the model at `deployment/saved_models/checkpoint.pth`.
4. **Run the API locally:**
   ```
   uvicorn deploy_endpoint:app --reload
   ```
5. **Test the API:**
   - Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
   - Use the `/predict/` endpoint to upload an `.mp4` file and get predictions.

## Deployment on Render

1. **Push your code to GitHub.**
2. **Create a new Web Service on [Render](https://render.com/):**
   - Connect your repo.
   - Render will use `render.yaml` for configuration.
   - The API will be available at your Render URL (e.g., `https://your-app.onrender.com`).

## Notes

- The `dataset/` folder is ignored in git and not needed for inference/deployment.
- For training, see scripts in the `training/` folder.
- The API is CPU-friendly, but for large models or faster inference, GPU is recommended (paid on Render).

## License

This project is for research and educational purposes. See `LICENSE` if present.