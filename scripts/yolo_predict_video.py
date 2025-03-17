from ultralytics import YOLO
import os
from tqdm import tqdm

best_cycle = 9
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{best_cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

vids_dir = "/home/jalal/projects/data/neurocig/vids/processed"
output_dir = f"/home/jalal/projects/data/neurocig/vids/results/cycle_{best_cycle}/annotated"

track = True
if track:
    output_dir = output_dir + '_tracked'


for vid_name in tqdm(os.listdir(vids_dir)):
    if vid_name.endswith('.mp4'):
        if track:
            # Tracking with BotSort
            model.track(os.path.join(vids_dir, vid_name),
            stream_buffer=True,
            save=True,
            save_txt=True,
            max_det=5,
            project=output_dir,
            name=vid_name.removesuffix('.mp4')
        )
        else:
            model.predict(os.path.join(vids_dir, vid_name),
                stream_buffer=True,
                save=True,
                save_txt=True,
                max_det=5,
                project=output_dir,
                name=vid_name.removesuffix('.mp4')
            )
