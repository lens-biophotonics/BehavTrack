# frame_sampling.py
"""
This module contains the core logic for stratified frame selection from multiple videos.
It performs the following main steps:

1. Feature Extraction:
   - Reads frames from each video at specified intervals (frame_skip).
   - Converts each frame to grayscale, applies Gaussian blur, and extracts a normalized histogram of intensities.
   - These histograms serve as simple global features representing the frame content.

2. Clustering:
   - All extracted frame features (histograms) from all videos are combined into a single dataset.
   - K-means clustering partitions these frames into k clusters based on their feature similarity.

3. Stratified Sampling:
   - Given a desired number of frames (n) to select in total, frames are sampled from each cluster proportionally to 
     that cluster's size.
   - This ensures that the chosen frames represent the overall distribution of frame types (clusters) present in the data.

4. Output:
   - The function returns a dictionary mapping each video filename to the selected frame indices and their respective clusters.
   - This result can be used downstream for annotation, tracking, and behavioral analysis.

Future Integrations:
- Annotation: After obtaining representative frames, users or automated tools can annotate frames (e.g., label mouse behaviors).
- Tracking: Integrate object detection or pose estimation models to track mouse positions and poses over selected frames.
- Behavioral Analysis: Use the selected frames and their annotations/tracking data to classify behaviors or identify patterns.

To integrate such future functionalities:
- Consider adding functions that load pre-trained models, run inference on frames, and store results.
- Implement feature extraction methods that incorporate detection/tracking results as additional frame-level features.
- Expand the clustering or sampling steps to consider behavioral states or previously defined annotations.
"""

import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict

def extract_histogram(frame, blur_ksize=(5, 5)):
    """
    Extract a normalized intensity histogram from a frame.
    
    Steps:
    - Convert frame to grayscale.
    - Apply Gaussian blur to reduce noise and small detail.
    - Compute a histogram of pixel intensities.
    - Normalize and flatten the histogram to a 1D vector.

    Parameters:
    - frame (numpy.ndarray): A single video frame in BGR format (as provided by OpenCV).
    - blur_ksize (tuple): Kernel size for GaussianBlur.

    Returns:
    - hist (numpy.ndarray): A 1D normalized histogram representing the frame's intensity distribution.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def get_video_frame_features(video_path, frame_skip=30, blur_ksize=(5,5)):
    """
    Extract features from a video by sampling frames at intervals defined by frame_skip.

    For every 'frame_skip'-th frame:
    - Capture the frame.
    - Extract a histogram-based feature representation.

    Parameters:
    - video_path (str): Path to the input video file.
    - frame_skip (int): Interval at which frames are sampled. For example, frame_skip=30 means 
                        extract features from frame 0, 30, 60, ...
    - blur_ksize (tuple): Kernel size for GaussianBlur.

    Returns:
    - features (list of np.ndarray): A list of feature vectors for the sampled frames.
    - frame_indices (list of int): A list of frame indices (0-based) corresponding to each feature.
    """
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            hist = extract_histogram(frame, blur_ksize=blur_ksize)
            features.append(hist)
            frame_indices.append(idx)
        idx += 1
    cap.release()
    return features, frame_indices

def stratified_frame_selection(video_folder, n=50, k=5, frame_skip=30, blur_ksize=(5,5), progress_callback=None):
    """
    Perform stratified frame selection from a set of videos in a given folder.

    Steps:
    1. Identify all video files in the folder.
    2. Extract frame features at given intervals from each video.
    3. Combine all features and cluster them into k clusters using K-means.
    4. Determine how many frames to pick from each cluster to total n frames, 
       ensuring representation of the overall frame distribution.
    5. Randomly sample frames within each cluster and return the selection.

    Parameters:
    - video_folder (str): Path to the folder containing video files.
    - n (int): Total number of frames to select across all videos.
    - k (int): Number of clusters for stratified sampling.
    - frame_skip (int): Extract features from every 'frame_skip'-th frame.
    - blur_ksize (tuple): Kernel size for Gaussian blur applied before histogram extraction.
    - progress_callback (callable): A function or lambda that takes an integer (0-100) to update progress.

    Returns:
    - result (dict):
        {
            'video_file_name': {
                'indices': [list_of_selected_frame_indices],
                'clusters': [list_of_cluster_labels_for_each_selected_frame]
            },
            ...
        }

    Raises:
    - ValueError: If no video files are found or no frames are extracted.
    """
    # Gather video files from the given folder
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        raise ValueError("No video files found in the selected folder.")

    all_features = []
    video_map = []    # Keeps track of which video each frame came from
    frame_map = []    # Keeps track of frame indices within each video

    total_videos = len(video_files)
    # Extract features from each video
    for i, vid in enumerate(video_files, start=1):
        video_path = os.path.join(video_folder, vid)
        feats, f_indices = get_video_frame_features(video_path, frame_skip=frame_skip, blur_ksize=blur_ksize)
        if feats:
            all_features.extend(feats)
            video_map.extend([vid]*len(feats))
            frame_map.extend(f_indices)

        # Update progress for feature extraction (use half of the progress range)
        if progress_callback:
            progress_callback(int((i / total_videos) * 50))

    if not all_features:
        raise ValueError("No frames extracted. Check your frame_skip or videos.")

    all_features = np.array(all_features)

    # If fewer frames than requested n, just return them all without clustering
    if len(all_features) <= n:
        result = defaultdict(lambda: {"indices": [], "clusters": []})
        for vid, idx in zip(video_map, frame_map):
            result[vid]["indices"].append(idx)
            result[vid]["clusters"].append(None)
        return dict(result)

    # Clustering: K-means to partition frames into k groups
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(all_features)

    # Update progress after clustering (arbitrary choice, e.g., 75%)
    if progress_callback:
        progress_callback(75)

    # Count frames per cluster
    cluster_counts = np.bincount(cluster_labels)
    total_frames = len(all_features)

    # Allocate frames proportionally to cluster size
    frames_per_cluster = [max(1, int((count/total_frames)*n)) for count in cluster_counts]

    # Adjust if rounding does not sum up to n
    difference = n - sum(frames_per_cluster)
    while difference != 0:
        for i in range(k):
            if difference == 0:
                break
            if difference > 0:
                frames_per_cluster[i] += 1
                difference -= 1
            else:
                if frames_per_cluster[i] > 1:
                    frames_per_cluster[i] -= 1
                    difference += 1

    # Select frames from each cluster
    result = defaultdict(lambda: {"indices": [], "clusters": []})
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        np.random.shuffle(cluster_indices)
        chosen = cluster_indices[:frames_per_cluster[cluster_id]]
        for c in chosen:
            vid = video_map[c]
            fidx = frame_map[c]
            result[vid]["indices"].append(fidx)
            result[vid]["clusters"].append(cluster_id)

    # Sort frames by index within each video for clarity
    for vid in result:
        sorted_pairs = sorted(zip(result[vid]["indices"], result[vid]["clusters"]), key=lambda x: x[0])
        result[vid]["indices"], result[vid]["clusters"] = zip(*sorted_pairs)

    # Final progress update
    if progress_callback:
        progress_callback(100)

    return dict(result)
