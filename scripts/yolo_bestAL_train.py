from ultralytics import YOLO

prev_cycle = 7
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{prev_cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)

# Train the model with GPUs
# 100 - 500 frames -> epoch 50 batch 16
# frames > 500  - > epoch 70 batch 24 (memory constraints 24GB with 3090)
epochs = 70
batch = 24


new_cycle = prev_cycle + 1
name = f"cycle_{new_cycle}"

results = model.train(data="yolo_dataset.yaml", epochs=epochs, imgsz=640, batch=batch, save=True, resume=False, name=name, project='/home/jalal/projects/data/neurocig/yolo')