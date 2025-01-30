from ultralytics import YOLO

# Load a model
model = YOLO("YOLO11x-pose.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="yolo_dataset.yaml", epochs=10, imgsz=640)

model.export('torchscript')