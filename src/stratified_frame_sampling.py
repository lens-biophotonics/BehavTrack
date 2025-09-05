"""
BehavTrack — stratified_frame_sampling
======================================

Select a balanced, representative subset of frames for annotation.

This module:
1) Samples frames at a fixed interval per video.
2) Extracts normalized grayscale intensity histograms as features.
3) Clusters frames (GPU K-Means via cuML if available; falls back to sklearn).
4) Allocates selections proportional to cluster sizes (stratified sampling).
5) Saves chosen frames and a JSON index for downstream splits.

Key functions:
- extract_histogram(frame) -> feature vector (len=256)
- get_video_frame_features(video_path, ...) -> (features, frame_indices)
- stratified_frame_selection(video_folder, ...) -> {video: {indices, clusters}}
- load_frame(video_path, idx) -> np.ndarray
- save_frames(results, video_folder, output_folder="frames") -> writes .jpg + JSON

Use this to build an initial, diverse pool for manual annotation or AL cycles.

Last updated:
    on 05-09-2025 by:
        - Kartik M. Jalal
"""

import os
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import json

# Optional GPU stack: try CuPy + cuML; fall back to CPU (NumPy + scikit-learn)
try:
    import cupy as cp  # type: ignore
    from cuml.cluster import KMeans as cuKMeans  # type: ignore
    _GPU_OK = True
except Exception:
    cp = None  # type: ignore
    _GPU_OK = False

try:
    from sklearn.cluster import KMeans as skKMeans  # CPU fallback
except Exception:
    skKMeans = None  # type: ignore



def extract_histogram(frame: np.ndarray, blur_ksize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Description
    -----------
    Extracts a normalized grayscale intensity histogram from an image frame.
    The histogram captures brightness distribution and is useful for comparing
    frames (e.g., in stratified sampling).

    Inputs
    ------
    frame : np.ndarray
        Input image in OpenCV BGR format (H x W x 3).
    blur_ksize : tuple of int, default (5, 5)
        Kernel size for Gaussian blur (width, height).
        Helps reduce noise before histogram computation.

    Returns
    -------
    hist : np.ndarray
        1D array of length 256 containing the normalized intensity histogram.
        Values are scaled such that the L2 norm = 1.
    """

    # Convert BGR image to single-channel grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to suppress noise before histogram extraction
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)

    # Compute histogram: 256 bins, pixel intensity range [0, 256)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Normalize histogram (default: L2 normalization), then flatten to 1D
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def get_video_frame_features(
    video_path: str,
    frame_skip: int = 30,
    blur_ksize: Tuple[int, int] = (5, 5)
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Description
    -----------
    Extracts intensity-histogram features from a video by sampling frames at 
    regular intervals. Useful for stratified frame selection in active learning.

    Inputs
    ------
    video_path : str
        Path to the input video file.
    frame_skip : int, default 30
        Interval between sampled frames (e.g., 30 → take every 30th frame).
    blur_ksize : tuple of int, default (5, 5)
        Gaussian blur kernel size applied before histogram extraction.

    Returns
    -------
    features : list of np.ndarray
        List of normalized histograms (length 256) for each sampled frame.
    frame_indices : list of int
        Corresponding indices of sampled frames in the original video.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return [], []

    features = []       # Store extracted histograms
    frame_indices = []  # Store frame indices for reference
    idx = 0             # Current frame index

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Sample every `frame_skip` frames
        if idx % frame_skip == 0:
            hist = extract_histogram(frame, blur_ksize=blur_ksize)
            features.append(hist)
            frame_indices.append(idx)

        idx += 1

    cap.release()
    return features, frame_indices


def stratified_frame_selection(
    video_folder: str,
    n: int = 50,
    k: int = 5,
    frame_skip: int = 30,
    blur_ksize: Tuple[int, int] = (5, 5),
    progress_callback: Optional[Callable[[int], Any]] = None,
    random_state: Optional[int] = 42,
    use_gpu: bool = True,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Description
    -----------
    Samples frames from all videos in a folder, extracts per-frame intensity histograms,
    clusters them (K-Means), and selects a stratified subset of frames proportional to
    cluster sizes. Returns, per video, the selected frame indices and their cluster IDs.

    Inputs
    ------
    video_folder : str
        Directory containing videos (extensions: .mp4, .avi, .mov, .mkv).
    n : int, default 50
        Total number of frames to select across all videos.
    k : int, default 5
        Number of clusters for K-Means over frame features.
    frame_skip : int, default 30
        Sample every `frame_skip`-th frame when extracting features.
    blur_ksize : tuple of int, default (5, 5)
        Gaussian blur kernel size for histogram extraction.
    progress_callback : callable(int) | None, default None
        Optional callback to receive integer progress updates (0–100).
    random_state : int | None, default 42
        Seed for reproducible clustering & shuffling. Use None for stochastic runs.
    use_gpu : bool, default True
        Try GPU-accelerated K-Means (cuML) if available; falls back to CPU otherwise.

    Returns
    -------
    result : dict
        {
          "<video_file_name>": {
              "indices": [frame_idx, ...],   # sorted ascending
              "clusters": [cluster_id, ...]  # aligned with indices
          },
          ...
        }
    """

    # Gather video files
    video_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    if not video_files:
        raise ValueError("No video files found in the selected folder.")

    # For reproducibility (affects shuffling and sklearn KMeans init)
    rng = np.random.default_rng(random_state)

    all_features: List[np.ndarray] = []
    video_map: List[str] = []   # which video each sampled frame came from
    frame_map: List[int] = []   # within-video frame index for each sampled frame

    total_videos = len(video_files)
    for i, vid in enumerate(video_files, start=1):
        video_path = os.path.join(video_folder, vid)
        feats, f_indices = get_video_frame_features(
            video_path,
            frame_skip=frame_skip,
            blur_ksize=blur_ksize
        )
        if feats:
            all_features.extend(feats)
            video_map.extend([vid] * len(feats))
            frame_map.extend(f_indices)

        # Feature extraction progress ≈ 50%
        if progress_callback:
            progress_callback(int((i / total_videos) * 50))

    if not all_features:
        raise ValueError("No frames extracted. Check your frame_skip or videos.")

    # Convert features to array [num_samples, 256]
    all_features_np = np.asarray(all_features, dtype=np.float32)
    total_frames = all_features_np.shape[0]

    # If we have fewer frames than requested, return all frames (no clustering)
    if total_frames <= n:
        result = defaultdict(lambda: {"indices": [], "clusters": []})
        for vid, idx in zip(video_map, frame_map):
            result[vid]["indices"].append(idx)
            result[vid]["clusters"].append(None)  # no cluster labels in this shortcut case
        # Final progress
        if progress_callback:
            progress_callback(100)
        return dict(result)

    # Cap k to #samples to avoid errors (k-means requires k <= n_samples)
    k_eff = max(1, min(k, total_frames))

    # ---- K-Means clustering (GPU if available & requested; else CPU) ----
    if use_gpu and _GPU_OK:
        # cuML KMeans expects CuPy arrays
        all_features_gpu = cp.asarray(all_features_np)  # type: ignore
        km = cuKMeans(
            n_clusters=k_eff,
            random_state=random_state if random_state is not None else 0,
            n_init=10,
            max_iter=300,
            verbose=0,
        )
        cluster_labels_gpu = km.fit_predict(all_features_gpu)
        cluster_labels = cp.asnumpy(cluster_labels_gpu).astype(int).tolist()  # type: ignore
    else:
        if skKMeans is None:
            raise RuntimeError(
                "CPU KMeans unavailable (scikit-learn not installed), and GPU KMeans not usable."
            )
        km = skKMeans(
            n_clusters=k_eff,
            random_state=random_state,
            n_init="auto" if hasattr(skKMeans, "n_init") else 10,  # compat across sklearn versions
            max_iter=300,
            verbose=0,
        )
        cluster_labels = km.fit_predict(all_features_np).astype(int).tolist()

    # Clustering progress ≈ 75%
    if progress_callback:
        progress_callback(75)

    # ---- Proportional allocation of selections per cluster ----
    cluster_labels_arr = np.asarray(cluster_labels, dtype=int)
    cluster_counts = np.bincount(cluster_labels_arr, minlength=k_eff)

    # Compute initial allocation proportional to cluster sizes, ensure at least 1 each
    frames_per_cluster = [max(1, int((count / total_frames) * n)) for count in cluster_counts]

    # Adjust rounding so that the sum equals exactly n
    diff = n - sum(frames_per_cluster)
    # Distribute the remainder across clusters (positive or negative)
    while diff != 0:
        for cid in range(k_eff):
            if diff == 0:
                break
            if diff > 0:
                frames_per_cluster[cid] += 1
                diff -= 1
            else:
                if frames_per_cluster[cid] > 1:
                    frames_per_cluster[cid] -= 1
                    diff += 1

    # ---- Sample frames within each cluster ----
    result = defaultdict(lambda: {"indices": [], "clusters": []})
    for cid in range(k_eff):
        cluster_indices = np.where(cluster_labels_arr == cid)[0]

        # Shuffle cluster membership deterministically using rng
        # (np.random.shuffle is global; use permutation instead)
        perm = rng.permutation(cluster_indices.shape[0])
        cluster_indices = cluster_indices[perm]

        chosen = cluster_indices[:frames_per_cluster[cid]]  # may be fewer if cluster is tiny

        # Record chosen frames per original video
        for c in chosen:
            vid = video_map[c]
            fidx = frame_map[c]
            result[vid]["indices"].append(fidx)
            result[vid]["clusters"].append(int(cid))

    # Sort selections per video by frame index for nicer downstream UX
    for vid in list(result.keys()):
        combined = sorted(
            zip(result[vid]["indices"], result[vid]["clusters"]),
            key=lambda x: x[0]
        )
        if combined:
            indices, clusters = zip(*combined)
            result[vid]["indices"] = list(indices)
            result[vid]["clusters"] = list(clusters)
        else:
            result[vid]["indices"] = []
            result[vid]["clusters"] = []

    # Final progress
    if progress_callback:
        progress_callback(100)

    return dict(result)


def load_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """
    Description
    -----------
    Loads a single frame from a video file at the specified frame index.

    Inputs
    ------
    video_path : str
        Path to the video file.
    frame_index : int
        Zero-based index of the frame to load.

    Returns
    -------
    frame : np.ndarray or None
        The loaded frame in OpenCV BGR format (H x W x 3).
        Returns None if the frame could not be read (e.g., out of range or invalid file).
    """

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Could not open the file
        return None

    # Move the capture pointer to the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = cap.read()
    cap.release()

    # Return frame only if read succeeded
    if not ret:
        return None

    return frame


def save_frames(
    results: Dict[str, Dict[str, List[int]]],
    video_folder: str,
    output_folder: str = "frames"
) -> None:
    """
    Description
    -----------
    Saves frames selected from stratified sampling into disk as `.jpg` images,
    and writes metadata (video name, frame index, cluster ID, image path) 
    into a `frames_info.json` file.

    Inputs
    ------
    results : dict
        Output dictionary from `stratified_frame_selection()`.
        {
            "video_file.mp4": {
                "indices": [frame indices...],
                "clusters": [cluster ids...]
            },
            ...
        }
    video_folder : str
        Path to the folder containing source video files.
    output_folder : str, default "frames"
        Folder where extracted frames and `frames_info.json` will be stored.

    Returns
    -------
    None
        Saves `.jpg` frames and metadata JSON file to `output_folder`.
    """

    os.makedirs(output_folder, exist_ok=True)

    frames_info: List[Dict[str, Any]] = []
    all_items = []  # Flattened (video, frame_index, cluster) tuples

    for vid, data in results.items():
        for idx, cluster in zip(data["indices"], data["clusters"]):
            all_items.append((vid, idx, cluster))

    total = len(all_items)
    print(f"Saving {total} frames to '{output_folder}'...")

    for i, (vid, frame_idx, cluster) in enumerate(all_items, start=1):
        video_path = os.path.join(video_folder, vid)
        frame = load_frame(video_path, frame_idx)
        if frame is None:
            raise ValueError(f"Could not load frame {frame_idx} from {vid}.")

        # Create filename: "<video_base>_frame_<index>.jpg"
        base_name, _ext = os.path.splitext(vid)
        frame_filename = f"{base_name}_frame_{frame_idx}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)

        # Save frame as JPEG
        cv2.imwrite(frame_path, frame)

        # Append metadata for this frame
        frames_info.append({
            "video": vid,
            "frame_index": frame_idx,
            "cluster": cluster,
            "image_path": frame_path
        })

        # Log progress every 10 frames (or at the very end)
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] frames saved.")

    # Write metadata JSON file
    json_path = os.path.join(output_folder, "frames_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(frames_info, f, indent=4)
    print(f"Metadata saved to {json_path}.")
