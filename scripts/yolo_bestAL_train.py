from ultralytics import YOLO

prev_cycle = 8
model_path = f"/home/jalal/projects/data/neurocig/yolo/cycle_{prev_cycle}/weights/best.pt"
# Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)

# Train the model with GPUs
# 100 - 500 frames -> epoch 50 batch 16
# 500 - 1000 -> epoch 70 batch 18 
# 1000 - 1500 -> epoch 90 batch 20
# frames > 1500 -> epoch 110 batch 22
epochs = 90
batch = 20

new_cycle = prev_cycle + 1
name = f"cycle_{new_cycle}"

results = model.train(data="yolo_dataset.yaml", epochs=epochs, imgsz=640, batch=batch, save=True, resume=False, name=name, project='/home/jalal/projects/data/neurocig/yolo')