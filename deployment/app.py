import gradio as gr
from inference import model_fn, predict_fn
import os

model_dict = model_fn(".")

def predict(video):
    temp_path = "temp.mp4"
    try:
        # Gradio gives a file path for video input
        if isinstance(video, str) and os.path.exists(video):
            os.rename(video, temp_path)
        else:
            with open(temp_path, "wb") as f:
                f.write(video.read())
        input_data = {"video_path": temp_path}
        result = predict_fn(input_data, model_dict)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Video(type="filepath"),
    outputs="json",
    title="Video Sentiment Analysis",
    description="Upload an .mp4 video to get sentiment and emotion predictions for each utterance."
)

if __name__ == "__main__":
    demo.launch()
