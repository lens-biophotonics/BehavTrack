from ultralytics import YOLO
import os

best_cycle = 9
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{best_cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

vids_dir = "/home/jalal/projects/data/neurocig/vids/processed"
output_dir = f"/home/jalal/projects/data/neurocig/vids/results/cycle_{best_cycle}/annotated"

for vid_name in os.listdir(vids_dir):
    if vid_name.endswith('.mp4'):
        model.predict(os.path.join(vids_dir, vid_name),
            stream_buffer=True,
            save=True,
            save_txt=True,
            max_det=5,
            project=output_dir,
            name=vid_name.removesuffix('.mp4')
        )
