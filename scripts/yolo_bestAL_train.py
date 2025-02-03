from ultralytics import YOLO

# Load a model
model = YOLO("/home/jalal/projects/data/neurocig/yolo/yolo11x-pose/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="yolo_dataset.yaml", epochs=50, imgsz=640, batch=16, save=True, resume=True, project='/home/jalal/projects/data/neurocig/yolo')