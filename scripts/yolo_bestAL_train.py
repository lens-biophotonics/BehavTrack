from ultralytics import YOLO

model_path = "/home/jalal/projects/data/neurocig/yolo/cycle_1/weights/best.pt"
# Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# 100 - 500 frames -> epoch 50 batch 16
# 500 - 1000 frames - > epoch 70 batch 32
# 1000 - 5000 frames -> epoch 100 batch 64
# 5000 < frames -> epoch 200 / 300 batch 128
cycle = 1
name = f"cycle_{cycle}"

results = model.train(data="yolo_dataset.yaml", epochs=50, imgsz=640, batch=16, save=True, resume=False, name=name, project='/home/jalal/projects/data/neurocig/yolo')