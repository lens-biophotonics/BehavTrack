from ultralytics import YOLO
import os

model_path = "/home/jalal/projects/data/neurocig/yolo/cycle_1/weights/best.pt"
# Load a model
model = YOLO(model_path)  # pretrained YOLO11n model

predict_dir = "/home/jalal/projects/data/neurocig/yolo/predict"

predict_images = []

for p_image in os.listdir(predict_dir):
    if p_image.endswith('.jpg'):
        p_image_path = os.path.join(predict_dir, p_image)
        predict_images.append(p_image_path)

# Run batched inference on a list of images
model.predict(predict_images, save=True, save_txt=True, project=predict_dir, name='results')  # return a generator of Results objects
