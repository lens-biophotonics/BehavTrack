from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-pose.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="yolo_dataset.yaml", epochs=50, imgsz=640, batch=16, save=True, name="Initial_run", project='/home/jalal/projects/data/neurocig/yolo')