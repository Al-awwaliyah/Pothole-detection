import os
import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def detect(image, conf):
    results = model.predict(image, conf=conf)
    result_img = results[0].plot()
    count = len(results[0].boxes)
    return result_img, f"Found {count} pothole(s)"

demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="pil", label="Upload a road photo"),
            gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="Confidence threshold")],
    outputs=[gr.Image(label="Result"), gr.Textbox(label="Count")],
    title="AI-Based Pothole Detector",
    description="Upload a road image to detect potholes."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
