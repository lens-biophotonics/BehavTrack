from ultralytics import YOLO
import os

best_cycle = 9
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{best_cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

vids_dir = "/home/jalal/projects/data/neurocig/vids/processed"
output_dir = f"/home/jalal/projects/data/neurocig/vids/results/cycle_{best_cycle}/annotated_videos"

output_folder = 'noTrack'

# Run batched inference on a list of images
model.predict(vids_dir, stream=True, stream_buffer=True ,save=True, save_txt=True, max_det=5, project=output_dir)  # return a generator of Results objects
