"""
BehavTrack — active_learning
============================

Utilities for splits, metadata I/O, and YOLO label conversion.

This module supports the active-learning loop by:
- Loading/saving JSON metadata.
- Creating stratified splits (train/val, prediction, reserve).
- Copying frames into split folders.
- Converting annotation JSON <-> YOLO text labels (with keypoints).
- Preparing train/val samples (images + labels) for YOLO.

Key functions (selection):
- load_metadata(dir, name) / save_metadata(dir, name, data)
- perform_split(metadata, ratio) -> (active_learning, prediction)
- print_details(metadata, split_data) / get_cluster_ratio(data)
- copy_frames(split_tuple) / copy_frames_train_val(src, dst)
- get_image_size(path)
- create_yolo_bBox_labels(annos, w, h, class_id) -> list[str]
- save_yolo_label(dst_dir, frame_name, labels)
- prepare_train_val(t_v_dir, annotations_dir, frame_name, info)
- yolo_txt_to_annotation_json(txt, image_name, W, H, flags, ...) -> dict
- get_yolo_vid_detections_in_json(...): collect YOLO .txt detections per video

Use these helpers to orchestrate each AL cycle: split → annotate → convert →
train → predict → merge → repeat.

Last updated:
    on 05-09-2025 by:
        - Kartik M. Jalal
"""

import os
from typing import List, Tuple, Dict, Union, Any, Optional

import json
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter
from PIL import Image
import re


def load_metadata(source_dir: str, metadata_filename: str) -> Dict[str, Any]:
    """
    Description
    -----------
    Loads a JSON metadata file from the given directory.

    Inputs
    ------
    source_dir : str
        Path to the directory containing the metadata file.
    metadata_filename : str
        Name of the metadata file (e.g., "frames_info.json").

    Returns
    -------
    metadata : dict
        Parsed JSON contents as a Python dictionary.
    """

    metadata_path = os.path.join(source_dir, metadata_filename)

    # Open and parse JSON metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def perform_split(
    metadata: List[Dict[str, Any]], 
    split_ratio: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Description
    -----------
    Splits metadata entries into an active-learning set and a prediction set,
    using stratified sampling based on cluster IDs.

    Inputs
    ------
    metadata : list of dict
        List of frame metadata entries (each entry typically includes 
        'video', 'frame_index', 'cluster', 'image_path').
    split_ratio : float
        Proportion of data to assign to the prediction set.
        Must be between 0.0 and 1.0 (inclusive).

    Returns
    -------
    activeLearning_data : list of dict
        Subset of metadata used for training/validation (active learning).
    prediction_data : list of dict
        Subset of metadata set aside for predictions.
    """

    if split_ratio > 1.0 or split_ratio < 0.0:
        # Invalid ratio: return empty split
        return [], metadata

    # Collect cluster labels for stratified sampling
    clusters = [entry["cluster"] for entry in metadata]

    # Perform stratified split
    activeLearning_data, prediction_data = train_test_split(
        metadata,
        test_size=split_ratio,
        random_state=42,     # ensures reproducibility
        stratify=clusters    # preserve cluster distribution
    )

    return activeLearning_data, prediction_data


def get_cluster_ratio(data: List[Dict[str, Any]]) -> None:
    """
    Description
    -----------
    Computes and prints the distribution of frames across clusters.

    Inputs
    ------
    data : list of dict
        Metadata entries (each entry must contain a 'cluster' key).

    Returns
    -------
    None
        Prints the number and percentage of frames per cluster.
    """

    # Count frames per cluster
    cluster_counts = Counter(entry["cluster"] for entry in data)

    # Total frames in the dataset
    total_frames = len(data)

    # Print percentage distribution for each cluster
    for cluster_id, count in cluster_counts.items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        print(f"Cluster {cluster_id}: {count} frames ({percentage:.2f}%)")


def print_details(
    metadata: List[Dict[str, Any]],
    split_data: List[Tuple[Tuple[str, str], Tuple[List[Dict[str, Any]], Any]]]
) -> None:
    """
    Description
    -----------
    Prints dataset statistics: total number of frames and cluster distributions
    for the full metadata and for each split.

    Inputs
    ------
    metadata : list of dict
        Complete set of metadata entries (each entry must contain a 'cluster' key).
    split_data : list of tuples
        Each element is expected to be of the form:
            ((<label>, <split_name>), (<data, ...>))
        - <split_name> : str, name of the split (e.g., "train", "test")
        - <data>       : list of dict entries (frames in that split)

    Returns
    -------
    None
        Prints dataset statistics and cluster ratios to stdout.
    """

    print("#############")
    print(f"Total frames: {len(metadata)}")
    get_cluster_ratio(metadata)  # Print cluster distribution of the full dataset
    print("\t#############")

    for split in split_data:
        split_name = split[0][1]       # extract split label (e.g., "train" or "test")
        split_metadata = split[1][0]   # actual list of frame metadata entries

        # Compute proportion of this split relative to total metadata
        percentage = (len(split_metadata) / len(metadata)) * 100 if len(metadata) > 0 else 0

        print(f"{split_name} split: {len(split_metadata)} ({percentage:.2f}%)")
        get_cluster_ratio(split_metadata)  # Print cluster ratio within the split
        print("\t#############")


def save_metadata(
    output_dir: str, 
    metadata_filename: str, 
    metadata: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> None:
    """
    Description
    -----------
    Saves metadata as a JSON file in the specified output directory.

    Inputs
    ------
    output_dir : str
        Path to the folder where the metadata file will be saved.
    metadata_filename : str
        Name of the output JSON file (e.g., "frames_info.json").
    metadata : dict or list of dict
        Metadata object to serialize (typically a list of frame entries or a dict).

    Returns
    -------
    None
        Writes JSON metadata to disk.
    """

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, metadata_filename)

    # Write JSON file with pretty indentation
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")


def copy_frames(split: Tuple[Tuple[str, str], Tuple[List[Dict[str, Any]], bool], Tuple[str, str]]) -> None:
    """
    Description
    -----------
    Copies frames and their metadata for a given dataset split into the 
    specified output folder. 

    The split tuple is expected to contain:
      (
        (split_label, split_name),          # e.g., ("train", "train_metadata.json")
        (split_metadata, copy_images_flag), # list of frame metadata + whether to copy images
        (src_frames_dir, out_dir)           # source frames dir + target output dir
      )

    Inputs
    ------
    split : tuple
        Structured as:
        (
            (str, str), 
            (list of dict, bool), 
            (str, str)
        )
        Where:
        - split[0][0] = split label (subfolder name, e.g., "train")
        - split[0][1] = metadata filename (e.g., "train_metadata.json")
        - split[1][0] = list of metadata dicts (must include "image_path")
        - split[1][1] = bool, whether to copy images
        - split[2][0] = source frames directory
        - split[2][1] = output directory

    Returns
    -------
    None
        Saves metadata JSON in the output directory and copies frames 
        if the flag is True.
    """

    # Destination path for this split (e.g., "output/train")
    out_path = os.path.join(split[2][1], split[0][0])
    os.makedirs(out_path, exist_ok=True)

    # Save metadata JSON into the output directory
    save_metadata(split[2][1], split[0][1], split[1][0])

    # Copy image files if the flag is set
    if split[1][1]:
        for data in split[1][0]:
            frame_name = os.path.basename(data["image_path"])
            src_frame_path = os.path.join(split[2][0], frame_name)
            out_frame_path = os.path.join(out_path, frame_name)
            shutil.copy2(src_frame_path, out_frame_path)

    print(f"Copy complete for split '{split[0][1]}' with images = {split[1][1]}")


def copy_frames_train_val(currentFrame_path: str, destination_dir: str) -> None:
    """
    Description
    -----------
    Copies a single frame file into the training/validation directory.

    Inputs
    ------
    currentFrame_path : str
        Full path to the source frame image file.
    destination_dir : str
        Directory where the frame should be copied.
        Created if it does not exist.

    Returns
    -------
    None
        Copies the file to the destination directory.
    """

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Copy frame into destination directory (preserves metadata like timestamps)
    shutil.copy2(currentFrame_path, destination_dir)


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    Description
    -----------
    Retrieves the dimensions (width, height) of an image file.

    Inputs
    ------
    image_path : str
        Path to the image file (e.g., .jpg, .png, .tiff).

    Returns
    -------
    (width, height) : tuple of int
        Image width and height in pixels.
    """

    # Open image file in context manager (auto-closes after use)
    with Image.open(image_path) as img:
        return img.width, img.height
    

def create_yolo_bBox_labels(
    annotation_bBox_info: List[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    class_label: int
) -> List[str]:
    """
    Description
    -----------
    Converts bounding box + keypoint annotations into YOLO label format.
    Each line encodes:
        class_id, x_center_norm, y_center_norm, w_norm, h_norm, 
        followed by normalized keypoints (x, y, v).

    Inputs
    ------
    annotation_bBox_info : list of dict
        List of bounding box annotations. Each dict must contain:
        - 'bbox': { "x1": int, "y1": int, "x2": int, "y2": int }
        - 'keypoints': { <name>: (x, y, v), ... }
    frame_w : int
        Frame width in pixels.
    frame_h : int
        Frame height in pixels.
    class_label : int
        Class ID for YOLO (e.g., 0 for "mouse").

    Returns
    -------
    yolo_bBox_label : list of str
        Each element is a YOLO-format annotation string for one bounding box:
        "class_id x_center y_center w h kp1_x kp1_y kp1_v ..."
    """

    yolo_bBox_label: List[str] = []

    for bBox_info in annotation_bBox_info:
        bBox = bBox_info["bbox"]

        # Extract raw bounding box coords
        x1, y1 = bBox["x1"], bBox["y1"]
        x2, y2 = bBox["x2"], bBox["y2"]

        # Convert to width, height, and center (absolute coords)
        bBox_w = x2 - x1
        bBox_h = y2 - y1
        x_center = x1 + bBox_w / 2.0
        y_center = y1 + bBox_h / 2.0

        # Normalize all values to [0, 1] relative to frame size
        x_center_norm = x_center / frame_w
        y_center_norm = y_center / frame_h
        w_norm = bBox_w / frame_w
        h_norm = bBox_h / frame_h

        # Start YOLO line with class_id + bounding box info
        yolo_line = (
            f"{class_label} "
            f"{x_center_norm:.6f} {y_center_norm:.6f} "
            f"{w_norm:.6f} {h_norm:.6f}"
        )

        # Append normalized keypoints (x, y, v)
        for keypoint in bBox_info["keypoints"].values():
            kp_x = keypoint[0] / frame_w
            kp_y = keypoint[1] / frame_h
            kp_v = keypoint[2]  # usually 0 (invisible), 1 (visible), or 2 (labeled)

            yolo_line += f" {kp_x:.6f} {kp_y:.6f} {kp_v}"

        # Add complete line to output
        yolo_bBox_label.append(yolo_line)

    return yolo_bBox_label


def save_yolo_label(
    t_v_images_dir: str,
    frame_name: str,
    yolo_bBox_labels: List[str]
) -> None:
    """
    Description
    -----------
    Saves YOLO-format bounding box labels for a given frame.
    Each frame gets a `.txt` file with one line per bounding box.

    Inputs
    ------
    t_v_images_dir : str
        Directory where YOLO `.txt` annotation files will be saved.
        Created if it does not exist.
    frame_name : str
        Name of the frame image file (e.g., "video1_frame_123.jpg").
        The `.jpg` extension is replaced with `.txt`.
    yolo_bBox_labels : list of str
        List of YOLO annotation strings, one per bounding box:
        "class_id x_center y_center w h kp1_x kp1_y kp1_v ..."

    Returns
    -------
    None
        Writes a `.txt` file with YOLO labels.
    """

    # Ensure destination directory exists
    os.makedirs(t_v_images_dir, exist_ok=True)

    # Replace ".jpg" with ".txt" in the frame name for YOLO label file
    t_v_label_filename = os.path.join(
        t_v_images_dir,
        frame_name.replace(".jpg", ".txt")
    )

    # Write YOLO labels (each bbox on its own line)
    with open(t_v_label_filename, "w", encoding="utf-8") as txt_out:
        txt_out.write("\n".join(yolo_bBox_labels))


def prepare_train_val(
    t_v_dir: str,
    annotations_dir: str,
    frame_name: str,
    annotation_info: List[Dict[str, Any]]
) -> None:
    """
    Description
    -----------
    Prepares training/validation data for YOLO by copying the frame image,
    generating YOLO-format bounding box + keypoint labels, and saving them
    alongside the image.

    Inputs
    ------
    t_v_dir : str
        Directory where the training/validation images and labels are stored.
    annotations_dir : str
        Directory containing original annotated frames.
    frame_name : str
        Name of the frame file (e.g., "video1_frame_123.jpg").
    annotation_info : list of dict
        Annotation data for the frame, with bounding boxes and keypoints.
        Expected format matches `create_yolo_bBox_labels`:
        [
            {
                "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                "keypoints": {<name>: (x, y, v), ...}
            },
            ...
        ]

    Returns
    -------
    None
        Copies the frame to the target directory and writes the `.txt` YOLO label file.
    """

    # Path to the annotated frame image
    currentFrame_path = os.path.join(annotations_dir, frame_name)

    # Copy the frame into the train/val directory
    copy_frames_train_val(currentFrame_path, t_v_dir)

    # Get image dimensions (needed for YOLO normalization)
    frame_w, frame_h = get_image_size(currentFrame_path)

    # Class label (currently fixed as 0 for "mouse")
    mouse_class_label = 0

    # Convert annotations to YOLO format
    yolo_bBox_labels = create_yolo_bBox_labels(
        annotation_info, frame_w, frame_h, mouse_class_label
    )

    # Save YOLO labels as .txt file next to the frame
    save_yolo_label(t_v_dir, frame_name, yolo_bBox_labels)


def is_point_in_bBox(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    """
    Description
    -----------
    Checks whether a given point lies inside (or on the edge of) 
    an axis-aligned bounding box.

    Inputs
    ------
    x, y : float
        Coordinates of the point to check.
    x1, y1 : float
        Top-left corner of the bounding box.
    x2, y2 : float
        Bottom-right corner of the bounding box.

    Returns
    -------
    inside : bool
        True if the point lies within the bounding box (inclusive of edges),
        otherwise False.
    """

    return (x1 <= x <= x2) and (y1 <= y <= y2)


def yolo_txt_to_annotation_json(
    txt_path: str,
    image_filename: str,   # e.g., "frame_001.jpg"
    image_width: int,
    image_height: int,
    manually_annotated_flag: bool,
    visible_percentage: float,
    keypoint_names: Optional[List[str]] = None,
    tracked: bool = False
) -> Dict[str, List[Dict]]:
    """
    Description
    -----------
    Converts a YOLO-style `.txt` annotation file (bounding box + keypoints) 
    into the internal JSON annotation format.

    Inputs
    ------
    txt_path : str
        Path to the YOLO `.txt` label file.
    image_filename : str
        Name of the associated image file (e.g., "frame_001.jpg").
    image_width : int
        Width of the image in pixels (needed to denormalize YOLO values).
    image_height : int
        Height of the image in pixels.
    manually_annotated_flag : bool
        Flag indicating whether the frame was manually annotated.
    visible_percentage : float
        Threshold for marking a keypoint as visible (v > threshold → visible).
    keypoint_names : list of str, optional
        Names of keypoints in order. Defaults to ["nose", "earL", "earR", "tailB"].
    tracked : bool
        If True, output uses {frame: {id: annotation}}, otherwise {frame: [annotation,...]}.
    Returns
    -------
    annotations : dict
        Dictionary with the structure:
        {
          "image_filename.jpg": [
            {
              "bbox": {"x1":..., "y1":..., "x2":..., "y2":...},
              "keypoints": { "nose": [...], "earL": [...], ... },
              "mAnnotated": bool
            },
            ...
          ]
        }
        or, if tracked=True:
        {
          "image_filename.jpg": {
            id:
            {
              "bbox": {"x1":..., "y1":..., "x2":..., "y2":...},
              "keypoints": { "nose": [...], "earL": [...], ... },
              "mAnnotated": bool
            },
            ...
          }
        }
    """
    if keypoint_names is None:
        keypoint_names = ["nose", "earL", "earR", "tailB"]

    # For tracked=False → list; for tracked=True → dict keyed by track id
    annotations: Dict[str, Any] = {image_filename: {} if tracked else []}

    # Read YOLO label file
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        # First 5 tokens: class_id, x_center, y_center, w, h
        class_id   = int(tokens[0])
        x_center_n = float(tokens[1])
        y_center_n = float(tokens[2])
        w_n        = float(tokens[3])
        h_n        = float(tokens[4])

        # Denormalize bounding box
        x_center = x_center_n * image_width
        y_center = y_center_n * image_height
        w        = w_n * image_width
        h        = h_n * image_height

        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # Skip degenerate bounding boxes
        if (x1 == x2 or y1 == y2):
            continue

        # Parse keypoints (each has x, y, v → 3 tokens)
        keypoints_dict = {}
        num_kpts = len(keypoint_names)

        for i in range(num_kpts):
            x_kpt_n = float(tokens[5 + 3 * i])
            y_kpt_n = float(tokens[5 + 3 * i + 1])
            v_kpt   = float(tokens[5 + 3 * i + 2])

            # Denormalize keypoint
            x_kpt = x_kpt_n * image_width
            y_kpt = y_kpt_n * image_height

            # Keep only keypoints inside the bounding box
            if not is_point_in_bBox(x_kpt, y_kpt, x1, y1, x2, y2):
                continue

            kpt_name = keypoint_names[i]
            # YOLO visibility values are mapped to 1 (present) or 2 (fully visible)
            keypoints_dict[kpt_name] = [
                x_kpt,
                y_kpt,
                2 if v_kpt > visible_percentage else 1
            ]

        annotation = {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "keypoints": keypoints_dict,
            "mAnnotated": manually_annotated_flag
        }

        if tracked:
            # For 4 kpts: track id should be at token index 17
            tracking_id = int(tokens[17])
            annotations[image_filename].update({f"{tracking_id}": annotation})
        else:
            annotations[image_filename].append(annotation)

    return annotations


def get_yolo_vid_detections_in_json(
    video_name: str,
    video_predicted_labels_path: str,
    frame_idx_regex: re.Pattern,
    video_width: int,
    video_height: int,
    visible_percentage: float,
    keypoint_names: List[str],
) -> Dict[str, Any]:
    """
    Description
    -----------
    Loads YOLO prediction `.txt` files for a single video, converts them into the
    internal JSON annotation format (per-frame detections), and returns a dictionary
    keyed by frame indices.

    Inputs
    ------
    video_name : str
        Name of the video (used for logging).
    video_predicted_labels_path : str
        Path to the directory containing YOLO `.txt` label files for this video.
    frame_idx_regex : re.Pattern
        Regex pattern to extract the frame index from label filenames.
        Must capture the index as group(1).
    video_width : int
        Width of the original video (pixels).
    video_height : int
        Height of the original video (pixels).
    visible_percentage : float
        Threshold for classifying a keypoint as visible (v > threshold → 2 else 1).
    keypoint_names : list of str
        Ordered names of keypoints (e.g., ["nose", "earL", "earR", "tailB"]).

    Returns
    -------
    detections : dict
        Per-frame detections in the format:
        {
          "frame_index_str": {
            "id_str": {
              "bbox": {...},
              "keypoints": {...},
              "mAnnotated": bool
            },
            ...
          },
          ...
        }
    """

    detections: Dict[str, Any] = {}

    # Iterate over all predicted YOLO label files for this video
    for predicted_label in os.listdir(video_predicted_labels_path):
        if not predicted_label.endswith(".txt"):
            continue  # skip non-label files

        # Extract the frame index from the filename using the regex
        m = frame_idx_regex.search(predicted_label)
        if not m:
            # Skip files that do not match the expected pattern
            continue

        frame_index = m.group(1)  # e.g., "123" from "..._123.txt"

        # Build the path to the label file
        text_label_path = os.path.join(video_predicted_labels_path, predicted_label)

        try:
            # Convert YOLO txt → internal annotation format
            frame_detections, valid = yolo_txt_to_annotation_json(
                txt_path=text_label_path,
                image_filename=frame_index,
                image_width=video_width,
                image_height=video_height,
                mAnnotated_flag=False,
                visiblePercentage=visible_percentage,
                keypoint_names=keypoint_names,
                tracked=True
            )
        except Exception as e:
            print(f"Error parsing {predicted_label} for {video_name}: {e}")
            continue

        # Skip frames where parsing failed or no detections were found
        if not valid or len(frame_detections) < 1:
            print(f"No detections in {predicted_label} for {video_name}")
            continue

        # Merge this frame's detections into the overall dict
        detections.update(frame_detections)

    # Sort dictionary by numeric frame index to keep frames ordered
    detections = dict(sorted(detections.items(), key=lambda kv: int(kv[0])))

    return detections