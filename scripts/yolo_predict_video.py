from ultralytics import YOLO
import os

cycle = 5
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

predict_dir = "/home/jalal/projects/data/neurocig/vids/processed"
output_dir = "/home/jalal/projects/data/neurocig/vids/results"

# predict_images = []

# for p_image in os.listdir(predict_dir):
#     if p_image.endswith('.jpg'):
#         p_image_path = os.path.join(predict_dir, p_image)
#         predict_images.append(p_image_path)

# Run batched inference on a list of images
model.track(os.path.join(predict_dir, "Gabbia2-D6-eCig(1)-pre.mp4"), stream_buffer=True ,save=True, save_txt=True, max_det=5, project=predict_dir, name='test2_track')  # return a generator of Results objects
