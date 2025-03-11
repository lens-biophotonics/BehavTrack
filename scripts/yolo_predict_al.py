from ultralytics import YOLO
import os

cycle = 10
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

predict_dir = "/home/jalal/projects/data/neurocig/yolo/predict"

# Run batched inference on a list of images
model.predict(predict_dir, save=True, save_txt=True, max_det=5, project=predict_dir, name='results')  # return a generator of Results objects
