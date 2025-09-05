"""
BehavTrack — package initializer
===============================

Convenience imports that expose the main pipeline entry points from submodules.

From here you can do:
    from src import (
        pre_process_video,
        stratified_frame_selection, save_frames,
        track, overlay_annotations_on_video,
        train_yolo, predict_frames, predict_videos,
        get_video_resolution, get_min_max_frame_rate, get_min_max_resolution, get_nb_frames,
        perform_split, load_metadata, print_details, copy_frames, prepare_train_val,
        get_image_size, yolo_txt_to_annotation_json, save_metadata, get_yolo_vid_detections_in_json
    )

This makes notebooks and scripts cleaner by centralizing imports.


Last updated:
    on 05-09-2025 by:
        - Kartik M. Jalal
"""

from .pre_process import (
    pre_process_video
)
from .stratified_frame_sampling import (
    stratified_frame_selection,
    save_frames
)
from .custom_tracking import (
    track,
    overlay_annotations_on_video
)
from .yolo import (
    train_yolo,
    predict_frames,
    predict_videos
)
from .data import (
    get_video_resolution,
    get_min_max_frame_rate,
    get_min_max_resolution,
    get_nb_frames
)
from .active_learning import (
    perform_split,
    load_metadata,
    print_details,
    copy_frames,
    prepare_train_val,
    get_image_size,
    yolo_txt_to_annotation_json,
    save_metadata,
    get_yolo_vid_detections_in_json
)



__all__ = [
    ## From pre_process.py
    "pre_process_video",
    ## From stratified_frame_sampling.py
    "stratified_frame_selection",
    "save_frames",
    ## From custom_tracking.py
    "track",
    "overlay_annotations_on_video",
    ## From yolo.py
    "train_yolo",
    "predict_frames",
    "predict_videos",
    ## From data.py
    "get_video_resolution",
    "get_min_max_frame_rate",
    "get_min_max_resolution",
    "get_nb_frames",
    ## From active_learning.py
    "perform_split",
    "load_metadata",
    "print_details",
    "copy_frames",
    "prepare_train_val",
    "get_image_size",
    "yolo_txt_to_annotation_json",
    "save_metadata",
    "get_yolo_vid_detections_in_json"
]

