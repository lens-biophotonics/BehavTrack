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
    

def save_metadata(output_dir, metadata_filename, metadata):
    metadata_outFilePath = os.path.join(output_dir, metadata_filename)

    with open(metadata_outFilePath, 'w') as f:
        json.dump(metadata, f, indent=4)


def copyframes(split):
    # destination path for the split frame
    out_path = os.path.join(split[2][1], split[0][0])
    os.makedirs(out_path, exist_ok=True)
    # save the metadata at the output
    save_metadata(split[2][1], split[0][1], split[1][0])

    if split[1][1]:
        for data in split[1][0]:
            frame_name = os.path.basename(data['image_path'])
            # src frame path
            src_framePath = os.path.join(split[2][0], frame_name)
            out_framePath = os.path.join(out_path, frame_name)
            # copy
            shutil.copy2(src_framePath, out_framePath)

    print(f"Copy complete for the split {split[0][1]} with images - {split[1][1]}")


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height
    

def isPointInBBox(x, y, x1, y1, x2, y2):
  return (
    x >= x1 and x <= x2 and
    y >= y1 and y <= y2
  )


def yolo_txt_to_annotation_json(
    txt_path, 
    image_filename,   # "image_filename.jpg"
    image_width, 
    image_height,
    mAnnotated_flag,
    visiblePercentage,
    keypoint_names=None
):
    """
    Reads a YOLO-like .txt (with bbox + 4 keypoints in normalized coords),
    and returns a dictionary in the original annotation style:

    {
      "image_filename": [
        {
          "bbox": {"x1":..., "y1":..., "x2":..., "y2":...},
          "keypoints": {
            "nose":  [...],
            "earL":  [...],
            "earR":  [...],
            "tailB": [...]
          }
        },
        ...
      ]
    }
    """
    if keypoint_names is None:
        # You can change the order or number of keypoints as needed:
        keypoint_names = ["nose", "earL", "earR", "tailB"]

    annotations = {image_filename: []}

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        # The first 5 tokens are class_id, x_center, y_center, w, h
        class_id    = int(tokens[0])
        x_center_n  = float(tokens[1])
        y_center_n  = float(tokens[2])
        w_n         = float(tokens[3])
        h_n         = float(tokens[4])

        # Denormalize bounding box
        x_center = x_center_n * image_width
        y_center = y_center_n * image_height
        w        = w_n * image_width
        h        = h_n * image_height

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        if (x1 == x2 or y1 == y2):
            continue

        # Next tokens: each keypoint has x_kpt_n, y_kpt_n, v_kpt
        # For 4 keypoints, that's 12 tokens, starting at index = 5
        keypoints_dict = {}
        num_kpts = len(keypoint_names)
        
        # i.e. for 4 keypoints, range(4) => 0..3
        for i in range(num_kpts):
            x_kpt_n = float(tokens[5 + 3*i])
            y_kpt_n = float(tokens[5 + 3*i + 1])
            v_kpt   = float(tokens[5 + 3*i + 2])

            # denormalize
            x_kpt = x_kpt_n * image_width
            y_kpt = y_kpt_n * image_height

            if not(isPointInBBox(x_kpt, y_kpt, x1, y1, x2, y2)):
                continue
            
            kpt_name = keypoint_names[i]
            
            keypoints_dict[kpt_name] = [x_kpt, y_kpt, 2 if v_kpt > visiblePercentage else 1]

        annotations[image_filename].append({
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "keypoints": keypoints_dict,
            "mAnnotated": mAnnotated_flag
        })

    return annotations



# Main function

def main():
    # dirs' path
    source_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/frames"
    activeLearning_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/stratifySplit_frames/activeLearning"
    predict_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/stratifySplit_frames/activeLearning/predict"

    # path to manual annotated frames and its json
    annotations_dir = os.path.join(activeLearning_dir, "annotations")
    annotation_json = "annotation.json"


    predict_metadata = load_metadata(activeLearning_dir, "predict.json")
    predict_metadata
    annotation_metadata = load_metadata(activeLearning_dir, "annotations_metadata.json")
    annotation_metadata.extend(predict_metadata)
    split_data_combine = (
            ('annotations', 'annotations_metadata.json'),
            (annotation_metadata, True),
            (source_dir, activeLearning_dir)
        )

    copyframes(split_data_combine)


    combinedAnnotated_json = load_metadata(annotations_dir, annotation_json)
    mAnnotated_flag = False
    visiblePercentage = 0.85
    for label in os.listdir(predict_dir):
        if label.endswith(".txt"):
            label_path = os.path.join(predict_dir, label)

            image_name = label.replace('txt', 'jpg')
            img_path = os.path.join(predict_dir, image_name)
            img_w, img_h = get_image_size(img_path)

            predictions = yolo_txt_to_annotation_json(label_path, image_name,img_w, img_h, mAnnotated_flag, visiblePercentage, ["nose", "earL", "earR", "tailB"])
            combinedAnnotated_json.update(predictions)

    save_metadata(annotations_dir, "annotation.json", combinedAnnotated_json)