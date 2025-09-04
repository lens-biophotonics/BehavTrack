import os
from typing import Any, Optional

from ultralytics import YOLO
from tqdm import tqdm

def train_yolo(
    model_path: str,
    yolo_dataset_yaml: str,
    epochs: int,
    batch: int,
    cycle_name: str,
    output_dir: str,
    imgsz: int = 640,
    save_model: bool = True,
    resume_cycle: bool = False
) -> Any:
    """
    Description
    -----------
    Trains an Ultralytics YOLO model on the provided dataset configuration.

    Inputs
    ------
    model_path : str
        Path to a YOLO model or checkpoint (e.g., "yolo11n.pt" or "runs/.../weights/best.pt").
    yolo_dataset_yaml : str
        Path to a dataset YAML file defining train/val/test image/label directories and class names.
    epochs : int
        Number of training epochs.
    batch : int
        Batch size for training.
    cycle_name : str
        Name of this training run (used as the subfolder under `output_dir`).
    output_dir : str
        Root output directory for training artifacts (Ultralytics will create `output_dir/cycle_name`).
    imgsz : int, default 640
        Square image size used for training/validation.
    save_model : bool, default True
        If True, save checkpoints and final weights.
    resume_cycle : bool, default False
        If True, resume training from the last checkpoint in this project/name.

    Returns
    -------
    results : Any
        Ultralytics `Results`/training summary object returned by `model.train()`.
    """

    # Load YOLO model/weights (can be a base model or a prior run's checkpoint)
    model = YOLO(model_path)


    # Kick off training. Ultralytics will:
    # - auto-detect device (GPU if available),
    # - create `output_dir/cycle_name`,
    # - log metrics and save artifacts (labels, weights, plots) when `save_model=True`.
    results = model.train(
        data=yolo_dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        save=save_model,
        resume=resume_cycle,
        name=cycle_name,
        project=output_dir
    )

    return results


def predict_frames(
    model_path: str,
    input_dir: str,
    output_dir: str,
    save: bool = True,
    save_txt: bool = True,
    max_det: int = 5,
    task_name: str = "prediction_frames_results"
) -> Any:
    """
    Description
    -----------
    Runs batched YOLO inference over a directory of frames and optionally
    saves visualized predictions and YOLO .txt labels.

    Inputs
    ------
    model_path : str
        Path to a YOLO model or checkpoint (e.g., "yolo11n.pt" or "runs/.../best.pt").
    input_dir : str
        Directory containing input frames/images to run inference on.
    output_dir : str
        Root directory to store prediction artifacts (images, labels, runs).
    save : bool, default True
        If True, saves rendered images with predictions overlaid.
    save_txt : bool, default True
        If True, saves YOLO-format label files for predictions.
    max_det : int, default 5
        Maximum number of detections per image.
    task_name : str, default "prediction_frames_results"
        Subfolder name under `output_dir` for this run.

    Returns
    -------
    results : Any
        Ultralytics prediction results object(s) returned by `model.predict()`.
    """

    # Ensure output directory exists (Ultralytics will create subfolder = task_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model/weights
    model = YOLO(model_path)

    # Run batched inference on the input directory
    results = model.predict(
        input_dir,
        save=save,
        save_txt=save_txt,
        max_det=max_det,
        project=output_dir,
        name=task_name
    )
    return results


def predict_videos(
    model_path: str,
    input_dir: str,
    output_dir: str,
    save: bool = True,
    save_txt: bool = True,
    max_det: int = 5,
    track: bool = False
) -> None:
    """
    Description
    -----------
    Runs YOLO inference (or tracking) on all `.mp4` videos in the given directory.
    Each video is processed independently and results are saved in subfolders 
    under the output directory.

    Inputs
    ------
    model_path : str
        Path to YOLO model or checkpoint (e.g., "yolo11n.pt" or "runs/.../best.pt").
    input_dir : str
        Directory containing `.mp4` video files for inference/tracking.
    output_dir : str
        Root directory where results (videos, labels, runs) will be stored.
    save : bool, default True
        If True, saves rendered videos with predictions overlaid.
    save_txt : bool, default True
        If True, saves YOLO-format `.txt` prediction files for each frame.
    max_det : int, default 5
        Maximum number of detections per frame.
    track : bool, default False
        If True, uses YOLO's `track()` method with BotSort for multi-object tracking.  
        If False, uses `predict()` for per-frame inference only.

    Returns
    -------
    None
        Saves predictions or tracking results to disk inside `output_dir`.
    """

    # Load YOLO model
    model = YOLO(model_path)

    # Iterate through all videos in input_dir
    for vid_name in tqdm(os.listdir(input_dir), desc="Processing videos"):
        if vid_name.endswith(".mp4"):
            video_path = os.path.join(input_dir, vid_name)
            run_name = vid_name.removesuffix(".mp4")  # folder name for this run

            if track:
                # Tracking with BotSort (keeps consistent IDs across frames)
                model.track(
                    video_path,
                    stream_buffer=True,
                    save=save,
                    save_txt=save_txt,
                    max_det=max_det,
                    project=output_dir,
                    name=run_name
                )
            else:
                # Frame-wise prediction only
                model.predict(
                    video_path,
                    stream_buffer=True,
                    save=save,
                    save_txt=save_txt,
                    max_det=max_det,
                    project=output_dir,
                    name=run_name
                )
