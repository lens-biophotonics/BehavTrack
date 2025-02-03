import os
import cv2
import json
import numpy as np
from collections import defaultdict
# from sklearn.cluster import KMeans
import cupy as cp
from cuml.cluster import KMeans


def extract_histogram(frame, blur_ksize=(5, 5)):
    """
    Extract a normalized intensity histogram from a frame (OpenCV BGR format).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def get_video_frame_features(video_path, frame_skip=30, blur_ksize=(5, 5)):
    """
    Extract features from a video by sampling frames at intervals defined by frame_skip.
    Returns a list of histogram features and their corresponding frame indices.
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


def stratified_frame_selection(
    video_folder,
    n=50,
    k=5,
    frame_skip=30,
    blur_ksize=(5, 5),
    progress_callback=None
):
    """
    Perform stratified frame selection from all videos in `video_folder`.
    Returns a dict:
        {
            'video_file_name': {
                'indices': [...],
                'clusters': [...]
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

    all_features = []
    video_map = []    # Which video each frame came from
    frame_map = []    # Frame indices within each video

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

        # Optional: update progress for feature extraction
        if progress_callback:
            # Suppose feature extraction counts for 50% of the total progress:
            progress_callback(int((i / total_videos) * 50))

    if not all_features:
        raise ValueError("No frames extracted. Check your frame_skip or videos.")

    all_features = np.array(all_features)

    # If fewer frames than requested, just return everything (no clustering)
    if len(all_features) <= n:
        result = defaultdict(lambda: {"indices": [], "clusters": []})
        for vid, idx in zip(video_map, frame_map):
            result[vid]["indices"].append(idx)
            result[vid]["clusters"].append(None)
        return dict(result)

    # # K-means clustering
    # kmeans = KMeans(n_clusters=k, random_state=42)
    # cluster_labels = kmeans.fit_predict(all_features)
    # GPU-based
    # Convert to GPU (CuPy) array
    all_features_gpu = cp.asarray(all_features)
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels_gpu = kmeans.fit_predict(all_features_gpu)
    # Convert back to CPU if you need Python ints
    cluster_labels = cluster_labels_gpu.get()

    # Optional: progress after clustering (e.g. 75%)
    if progress_callback:
        progress_callback(75)

    # Count frames per cluster
    cluster_counts = np.bincount(cluster_labels)
    total_frames = len(all_features)

    # Allocate frames proportionally to cluster size
    frames_per_cluster = [
        max(1, int((count / total_frames) * n))
        for count in cluster_counts
    ]

    # Adjust if rounding does not sum up to n
    difference = n - sum(frames_per_cluster)
    while difference != 0:
        for cluster_id in range(k):
            if difference == 0:
                break
            if difference > 0:
                frames_per_cluster[cluster_id] += 1
                difference -= 1
            else:
                # Only reduce if >1 (ensuring we don't go to zero)
                if frames_per_cluster[cluster_id] > 1:
                    frames_per_cluster[cluster_id] -= 1
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

    # Sort each video's frames by index
    for vid in result:
        combined = sorted(
            zip(result[vid]["indices"], result[vid]["clusters"]),
            key=lambda x: x[0]
        )
        indices, clusters = zip(*combined)
        result[vid]["indices"] = list(indices)
        result[vid]["clusters"] = list(clusters)

    # Optional: final progress
    if progress_callback:
        progress_callback(100)

    return dict(result)


def load_frame(video_path, frame_index):
    """
    Loads a single frame from the video at the given frame_index.
    Returns None if the frame couldn't be loaded.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def save_frames(results, video_folder, output_folder="frames"):
    """
    Given the stratified selection results, save each selected frame as a .jpg
    and write metadata to `frames_info.json`.

    :param results: dict from stratified_frame_selection()
    :param video_folder: base path of the videos
    :param output_folder: where to store the .jpg frames + JSON
    """
    os.makedirs(output_folder, exist_ok=True)

    frames_info = []
    # Flatten (video, frame_index, cluster) in a single list
    all_items = []
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

        # Define a unique filename
        base_name, _ext = os.path.splitext(vid)
        frame_filename = f"{base_name}_frame_{frame_idx}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)

        # Write the image
        cv2.imwrite(frame_path, frame)

        # Accumulate metadata
        frames_info.append({
            "video": vid,
            "frame_index": frame_idx,
            "cluster": cluster,
            "image_path": frame_path
        })

        # Optional progress logging
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] frames saved.")

    # Finally, write JSON
    json_path = os.path.join(output_folder, "frames_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(frames_info, f, indent=4)
    print(f"Metadata saved to {json_path}.")


def main():
    """
    Example main function that:
      1) Runs stratified frame selection.
      2) Saves the selected frames & metadata.
    """
    # -- 1. Set your parameters here --
    video_folder = "/home/jalal/projects/data/neurocig/vids/processed"
    n = 2000
    k = 8
    frame_skip = 50
    output_folder = "/home/jalal/projects/data/neurocig/frames"

    # -- 2. Run stratified frame selection --
    results = stratified_frame_selection(
        video_folder,
        n=n,
        k=k,
        frame_skip=frame_skip,
        blur_ksize=(5, 5),
        progress_callback=None  # or define a function for console progress updates
    )
    print("Stratified frame selection done.")
    print("Selected frames summary:")
    for vid, data in results.items():
        print(f"  Video: {vid}, Frames chosen: {len(data['indices'])}")

    # -- 3. Save the frames & metadata --
    save_frames(results, video_folder, output_folder)
    print("All done.")


if __name__ == "__main__":
    main()
