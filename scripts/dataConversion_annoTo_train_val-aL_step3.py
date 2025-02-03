# Libraries

import os
import shutil
import json
from PIL import Image

# Helper Functions

def load_metadata(source_dir, metadata_filename):
    metadata_filePath = os.path.join(source_dir, metadata_filename)

    with open(metadata_filePath, 'r') as f:
        return json.load(f)
    

def copyframes_train_val(currentFrame_path, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(currentFrame_path, destination_dir)


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height
    

def create_yolo_bBox_labels(annotation_bBox_info, frame_w, frame_h, class_label):
    yolo_bBox_label = []
    for bBox_info in annotation_bBox_info:
        bBox = bBox_info['bbox']

        x1, y1 = bBox["x1"], bBox["y1"]
        x2, y2 = bBox["x2"], bBox["y2"]

        # Convert to YOLO
        bBox_w = x2 - x1
        bBox_h = y2 - y1
        x_center = x1 + bBox_w / 2.0
        y_center = y1 + bBox_h / 2.0
        
        # Normalize
        x_center_norm = x_center / frame_w
        y_center_norm = y_center / frame_h
        w_norm = bBox_w / frame_w
        h_norm = bBox_h / frame_h
        
        # class_id, x_c, y_c, w, h
        yolo_line = f"{class_label} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        
        for keypoint in bBox_info['keypoints'].values():
            kp_x = keypoint[0]/frame_w
            kp_y = keypoint[1]/frame_h
            kp_v = keypoint[2]

            keypoint_line = f" {kp_x:.6f} {kp_y:.6f} {kp_v}"

            yolo_line += keypoint_line
        
        yolo_bBox_label.append(yolo_line)
        
    return yolo_bBox_label


def save_yolo_label(t_v_images_dir, frame_name, yolo_bBox_labels):
    os.makedirs(t_v_images_dir, exist_ok=True)
    t_v_label_filename = os.path.join(t_v_images_dir, frame_name.replace("jpg", "txt"))

    with open(t_v_label_filename, 'w') as txt_out:
        txt_out.write("\n".join(yolo_bBox_labels))


def prepare_train_val(t_v_dir, annotations_dir, frame_name, annotation_info):
    t_v_images_dir = os.path.join(t_v_dir, "images")
    currentFrame_path = os.path.join(annotations_dir, frame_name)
    copyframes_train_val(currentFrame_path, t_v_images_dir)

    frame_w, frame_h = get_image_size(currentFrame_path)
    mouse_class_label = 0
    yolo_bBox_labels = create_yolo_bBox_labels(annotation_info,  frame_w, frame_h, mouse_class_label)
    save_yolo_label(t_v_images_dir, frame_name, yolo_bBox_labels)


# Main()
def main():
    # dirs' path
    source_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/frames"
    output_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/stratifySplit_frames"
    activeLearning_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/stratifySplit_frames/activeLearning"


    train_dir = os.path.join(activeLearning_dir, "train")
    train_json = "train.json"
    val_dir = os.path.join(activeLearning_dir, "val")
    val_json = "val.json"

    # path to manual annotated frames and its json
    annotations_dir = os.path.join(activeLearning_dir, "annotations")
    annotation_json = "annotation.json"


    # manually annotated json
    mAnnotated_json = load_metadata(annotations_dir, annotation_json)

    # train and val frame metadata
    train_metadata = load_metadata(train_dir, train_json)
    val_metadata = load_metadata(val_dir, val_json)


    for frame_name, annotation_info in mAnnotated_json.items():
        for train_frame in train_metadata:
            if frame_name in train_frame['image_path']:
                prepare_train_val(train_dir, annotations_dir, frame_name, annotation_info)


        for val_frame in val_metadata:
            if frame_name in val_frame['image_path']:
                prepare_train_val(val_dir, annotations_dir, frame_name, annotation_info)



if __name__ == "__main__":
    main()