"""
BehavTrack — custom_tracking
============================

Identity tracking and visualization for YOLO detections on mouse videos.

This module takes per-frame YOLO detections (bbox + keypoints) and assigns
consistent IDs across frames. It blends bounding-box IoU and center distance,
plus keypoint similarity, and solves matches with the Hungarian algorithm.
It also handles frame gaps, ID reseeding, and a simple “stuck ID” release rule.
Optionally, it overlays tracked annotations onto video for QC.

Key functions:
- track(...): propagate stable IDs frame-to-frame using a cost function.
- overlay_annotations_on_video(...): render bboxes/keypoints/IDs into a video.

Typical use:
1) Run YOLO on videos to get frame-wise detections.
2) Call track(...) to assign stable IDs.
3) (Optional) Call overlay_annotations_on_video(...) to visualize results.


Last updated:
    on 05-09-2025 by:
        - Kartik M. Jalal
"""


import os
import cv2
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_bBox_xyxyc(bBox: dict) -> tuple:
    """
    Description
    -----------
    Convert a bounding box dictionary into a tuple of corner coordinates 
    and its center point.

    Inputs
    ------
    bBox : dict
        Dictionary with keys:
        - "x1": float, left (min) x-coordinate
        - "y1": float, top (min) y-coordinate
        - "x2": float, right (max) x-coordinate
        - "y2": float, bottom (max) y-coordinate

    Returns
    -------
    tuple
        (x1, y1, x2, y2, c_x, c_y) where:
        - x1, y1, x2, y2 : bounding box corners
        - c_x, c_y       : center coordinates of the box
    """
    c_x = 0.5 * (bBox["x1"] + bBox["x2"])
    c_y = 0.5 * (bBox["y1"] + bBox["y2"])

    return (bBox["x1"], bBox["y1"], bBox["x2"], bBox["y2"], c_x, c_y)


def get_keypoints_xyxyc(
    keypoints: Dict[str, List[float]],
    bBox: Dict[str, float],
    scale_factor: float = 0.02
) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """
    Description
    -----------
    Creates small bounding boxes around each keypoint relative to its parent 
    object bounding box. Returns both the box corners and the keypoint center.

    Inputs
    ------
    keypoints : dict
        Dictionary of keypoints in the format:
        {
            'nose'  : [x, y, visible_flag],
            'earL'  : [x, y, visible_flag],
            'earR'  : [x, y, visible_flag],
            'tailB' : [x, y, visible_flag]
        }
        - x, y : pixel coordinates of the keypoint
        - visible_flag : 0 (not labeled), 1 (labeled but not visible), 2 (labeled and visible)
    bBox : dict
        Parent bounding box with keys "x1", "y1", "x2", "y2".
    scale_factor : float, default 0.02
        Size of keypoint boxes as a fraction of the parent bbox width/height.

    Returns
    -------
    keypoints_xyxyc : dict
        Dictionary mapping each keypoint to a tuple:
        (x1, y1, x2, y2, c_x, c_y)
        where (x1,y1,x2,y2) are the corners of the small keypoint box and 
        (c_x,c_y) is the keypoint center.
    """

    keypoints_xyxyc: Dict[str, Tuple[float, float, float, float, float, float]] = {}

    # Parent bbox dimensions
    bBox_w = max(0.0, bBox["x2"] - bBox["x1"])
    bBox_h = max(0.0, bBox["y2"] - bBox["y1"])

    # Keypoint box dimensions relative to bbox size
    keypoint_bBox_w = scale_factor * bBox_w
    keypoint_bBox_h = scale_factor * bBox_h

    for keypoint, coordinates in keypoints.items():
        # Center point of the keypoint
        cx, cy = coordinates[0], coordinates[1]

        # Construct small box around the keypoint
        x1 = cx - (keypoint_bBox_w / 2)
        y1 = cy - (keypoint_bBox_h / 2)
        x2 = cx + (keypoint_bBox_w / 2)
        y2 = cy + (keypoint_bBox_h / 2)

        keypoints_xyxyc[keypoint] = (x1, y1, x2, y2, cx, cy)

    return keypoints_xyxyc


def get_iou(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float]
) -> float:
    """
    Description
    -----------
    Computes the Intersection-over-Union (IoU) between two bounding boxes.

    Inputs
    ------
    b1 : tuple of float
        Bounding box (x1, y1, x2, y2) in the same coordinate system.
    b2 : tuple of float
        Bounding box (x1, y1, x2, y2) in the same coordinate system.

    Returns
    -------
    iou : float
        IoU value in the range [0, 1].
        Returns 0 if boxes do not overlap or if union area is invalid.
    """
    # Intersection coordinates
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    # Clamp to non-negative
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    # Areas
    area1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
    union = area1 + area2 - inter

    if union < 1e-9:
        return 0.0
    return inter / union


def get_center_distance(
    b1: Tuple[float, float, float, float, float, float],
    b2: Tuple[float, float, float, float, float, float],
    abs_w: float,
    abs_h: float
) -> float:
    """
    Description
    -----------
    Computes the normalized Euclidean distance between the centers of two bounding boxes.

    Inputs
    ------
    b1, b2 : tuple of float
        Bounding boxes in (x1, y1, x2, y2, c_x, c_y) format.
    abs_w, abs_h : float
        Absolute width and height of the image (used for normalization).

    Returns
    -------
    distance : float
        Center distance normalized by the image diagonal. 
        Value is in [0, 1] if centers are within the image.
    """
    c1_x, c1_y = b1[4], b1[5]
    c2_x, c2_y = b2[4], b2[5]

    dx = c1_x - c2_x
    dy = c1_y - c2_y

    euclidean_distance = (dx**2 + dy**2) ** 0.5
    img_diagonal = (abs_w**2 + abs_h**2) ** 0.5 or 1.0  # avoid divide-by-zero

    return euclidean_distance / img_diagonal


def iou_keypoints(
    track: Dict[str, Tuple[float, float, float, float, float, float]],
    det: Dict[str, Tuple[float, float, float, float, float, float]]
) -> float:
    """
    Description
    -----------
    Computes the average IoU across overlapping keypoints between two objects.
    Each keypoint is represented as a small bounding box.

    Inputs
    ------
    track : dict
        Keypoints of the tracked object.
        { "nose": (x1, y1, x2, y2, c_x, c_y), ... }
    det : dict
        Keypoints of the detected object.
        { "nose": (x1, y1, x2, y2, c_x, c_y), ... }

    Returns
    -------
    score : float
        Weighted IoU score in [0, 1].
        = (fraction of overlapping keypoints) * (average IoU of those keypoints).
    """
    keypoints_present = 0
    iou_sum = 0.0

    for keypoint, coordinates in track.items():
        if keypoint not in det:
            continue
        # IoU only cares about the bbox corners
        iou_sum += get_iou(coordinates[:4], det[keypoint][:4])
        keypoints_present += 1

    if keypoints_present == 0:
        return 0.0

    avg_iou = iou_sum / keypoints_present
    return (keypoints_present / 4) * avg_iou


def center_distance_keypoints(
    track: Dict[str, Tuple[float, float, float, float, float, float]],
    det: Dict[str, Tuple[float, float, float, float, float, float]],
    abs_w: float,
    abs_h: float,
    penalty_per_missing: float = 10.0
) -> float:
    """
    Description
    -----------
    Computes an average normalized center distance between corresponding keypoints
    of two objects, with a penalty for missing keypoints.

    Inputs
    ------
    track : dict
        Keypoints of the tracked object.
        { "nose": (x1, y1, x2, y2, c_x, c_y), ... }
    det : dict
        Keypoints of the detected object.
        { "nose": (x1, y1, x2, y2, c_x, c_y), ... }
    abs_w, abs_h : float
        Image width and height used for normalization.
    penalty_per_missing : float, default 10.0
        Scaling factor for penalizing missing keypoints.
        (Note: current formula does not directly use this, see below.)

    Returns
    -------
    score : float
        Lower values mean closer matches.
        Returns 1.0 if no overlapping keypoints are present.
    """
    keypoints_present = 0
    total_distance = 0.0

    for keypoint, coordinates in track.items():
        if keypoint not in det:
            continue
        # Euclidean distance between keypoint centers (normalized)
        total_distance += get_center_distance(coordinates, det[keypoint], abs_w, abs_h)
        keypoints_present += 1

    missing_keypoints = 4 - keypoints_present

    if keypoints_present > 0:
        avg_distance = total_distance / keypoints_present
    else:
        # If no keypoints detected at all, return maximum penalty
        return 1.0

    # Current formula: balances avg distance and missing penalty
    total_score = (avg_distance + missing_keypoints) / (missing_keypoints + 1)
    return total_score


def compute_cost(
    track: Dict[str, Any],
    detection: Dict[str, Any],
    scale_factor: float,
    penalty_per_missing: float,
    abs_w: float,
    abs_h: float,
    alpha: float = 0.5,
    epsilon: float = 1e-6
) -> float:
    """
    Description
    -----------
    Computes a matching cost between a tracked object and a new detection by
    blending center-distance (lower is better) and IoU (higher is better) from
    both bounding boxes and keypoints. Lower cost ⇒ better match.

    Inputs
    ------
    track : dict
        Tracked object with keys: 'bbox' and 'keypoints'.
    detection : dict
        New detection with keys: 'bbox' and 'keypoints'.
    scale_factor : float
        Fraction of parent bbox size used to build small keypoint boxes.
    penalty_per_missing : float
        Passed through to keypoint center-distance (kept for compatibility).
    abs_w, abs_h : float
        Image width/height used for center-distance normalization.
    alpha : float, default 0.5
        Balance between IoU and center-distance terms (0..1).
    epsilon : float, default 1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    cost : float
        Normalized cost; smaller values indicate better matches.
    """

    # --- Bounding box terms ---
    track_b = get_bBox_xyxyc(track["bbox"])       # (x1,y1,x2,y2,cx,cy)
    det_b   = get_bBox_xyxyc(detection["bbox"])   # (x1,y1,x2,y2,cx,cy)

    iou_box   = get_iou(track_b[:4], det_b[:4])   # IoU uses corners only
    cdist_box = get_center_distance(track_b, det_b, abs_w, abs_h)

    # --- Keypoint terms ---
    track_k = get_keypoints_xyxyc(track["keypoints"], track["bbox"], scale_factor)
    det_k   = get_keypoints_xyxyc(detection["keypoints"], detection["bbox"], scale_factor)

    iou_k   = iou_keypoints(track_k, det_k)
    cdist_k = center_distance_keypoints(track_k, det_k, abs_w, abs_h, penalty_per_missing)

    # --- Blend into costs ---
    # Larger IoU should reduce cost; larger distance should increase cost.
    cdist_cost = (1.0 - alpha) * ((cdist_box + cdist_k) / 2.0)
    iou_cost   = alpha * ((iou_box + iou_k) / 2.0)

    # Final normalized cost; can be negative if IoU dominates distance (expected).
    return (cdist_cost - iou_cost) / (cdist_cost + iou_cost + epsilon)


def _make_id_list(max_ids: int) -> List[str]:
    """
    Description
    -----------
    Builds sequential string IDs: ["1", "2", ..., str(max_ids)].

    Inputs
    ------
    max_ids : int
        Maximum number of IDs to maintain.

    Returns
    -------
    ids : list of str
        List of ID strings in ascending order.
    """
    return [str(i) for i in range(1, max_ids + 1)]


def track(
    vid_name: str,
    nb_fames: int,
    detections: Dict[str, Dict[str, Any]],
    frames_skip_limit: int,
    scale_factor: float,
    penalty_per_missing: float,
    abs_w: float,
    abs_h: float,
    alpha: float,
    epsilon: float,
    cost_threshold: float = 0.5,
    release_id_at_value: int = 16,
    printing: bool = False,
    max_ids: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Description
    -----------
    Propagates consistent IDs across frames by matching detections between
    consecutive frames using a cost function (Hungarian assignment). Handles
    missing frames, new IDs, and "stuck" IDs with deterministic rules.

    Inputs
    ------
    vid_name : str
        Video name (for logging).
    nb_fames : int
        Total number of frames in the video.
    detections : dict
        Per-frame detections: { "frame_index(str)": { "id(str)": {bbox,keypoints,...}, ... }, ... }.
    frames_skip_limit : int
        Maximum gap allowed between indexed frames before re-seeding IDs.
    scale_factor : float
        Scale for keypoint boxes (fraction of bbox size) in matching.
    penalty_per_missing : float
        Penalty parameter forwarded to keypoint distance (compatibility).
    abs_w, abs_h : float
        Image width/height used for center-distance normalization.
    alpha : float
        Balance between IoU (bigger is better) and distance (smaller is better).
    epsilon : float
        Small constant to avoid division by zero in cost normalization.
    cost_threshold : float, default 0.5
        If the matched pair's cost ≤ threshold (or stuck condition holds), accept the match.
    release_id_at_value : int, default 16
        Frames after which a stuck ID is force-released.
    printing : bool, default False
        If True, prints case-by-case debugging info.
    max_ids : int, default 5
        Maximum number of concurrent track IDs (IDs are the strings "1"..str(max_ids)).

    Returns
    -------
    tracked : dict
        Updated `detections` with consistent ID assignment across frames.
    """

    id_list: List[str] = _make_id_list(max_ids)
    currentFrame_index = 1

    # Track stuck IDs: id -> (is_stuck, stuck_time)
    stuck_ids: Dict[str, Tuple[bool, int]] = {i: (False, 0) for i in id_list}

    # Ceiling for scanning missing indices: use actual max key present
    max_key = max((int(k) for k in detections.keys()), default=0)

    while currentFrame_index < nb_fames:
        nextFrame_index = currentFrame_index + 1

        # Find the next available frame index in detections (skip gaps)
        index_flag = f"{nextFrame_index}" in detections
        index_jump = False
        while not index_flag:
            if nextFrame_index > max_key:
                break
            if printing:
                print(f"Missing index {nextFrame_index} - {vid_name}")
            nextFrame_index += 1
            index_flag = f"{nextFrame_index}" in detections
            index_jump = True

        if not index_flag:
            break

        # If the gap is too large, jump and reseed IDs in that frame
        if index_jump and (nextFrame_index - currentFrame_index > frames_skip_limit):
            currentFrame_index = nextFrame_index
            available_ids = id_list.copy()
            temp_holder: Dict[str, Any] = {}
            for _, ann in detections[f"{currentFrame_index}"].items():
                if not available_ids:
                    break  # in case more detections than max_ids
                temp_holder[available_ids.pop()] = ann
            detections[f"{currentFrame_index}"] = temp_holder
            continue

        # Presence map: mouse_ID -> (exists_in_current, exists_in_next)
        valid_mouseId_presence: Dict[str, Tuple[bool, bool]] = {i: (False, False) for i in id_list}
        for cur_id, _ in detections[f"{currentFrame_index}"].items():
            if cur_id in valid_mouseId_presence:
                valid_mouseId_presence[cur_id] = (True, (cur_id in detections[f"{nextFrame_index}"]))

        # New IDs that appear in next frame but not in current
        new_nextFrame_mouseIds: List[str] = []
        for next_id, _ in detections[f"{nextFrame_index}"].items():
            if next_id in detections[f"{currentFrame_index}"]:
                continue
            new_nextFrame_mouseIds.append(next_id)

        # Determine unmatched current-frame IDs
        valid_toSkip_flag = True
        unmatched_currentFrame_Ids: List[str] = []
        for cur_id, flags in valid_mouseId_presence.items():
            if flags[1]:
                continue
            if flags[0]:  # present now, missing next
                unmatched_currentFrame_Ids.append(cur_id)
                valid_toSkip_flag = False

        # Case 1: perfect continuity (no new IDs, all current matched) → just advance
        if valid_toSkip_flag and (len(new_nextFrame_mouseIds) == 0):
            if printing:
                print(f"Case 1 {currentFrame_index} - {nextFrame_index}")
            currentFrame_index = nextFrame_index
            continue

        # Case 2: all current matched, but new IDs appear → assign free IDs
        elif valid_toSkip_flag and (len(new_nextFrame_mouseIds) > 0):
            if printing:
                print(f"Case 2 {currentFrame_index} - {nextFrame_index}")
            for new_id in new_nextFrame_mouseIds:
                for vid in id_list:
                    if valid_mouseId_presence[vid][0]:
                        continue
                    valid_mouseId_presence[vid] = (True, True)
                    detections[f"{nextFrame_index}"].update({
                        vid: detections[f"{nextFrame_index}"].pop(new_id)
                    })
                    if printing:
                        print(f"\t{vid} - {new_id}")
                    break
            currentFrame_index = nextFrame_index
            continue

        # Case 3: some current IDs missing in next and no new next IDs → propagate
        if (not valid_toSkip_flag) and (len(new_nextFrame_mouseIds) == 0):
            if printing:
                print(f"Case 3 {currentFrame_index} - {nextFrame_index}")
            for vid, flags in valid_mouseId_presence.items():
                if (not flags[0]) or flags[1]:
                    continue
                valid_mouseId_presence[vid] = (True, True)
                detections[f"{nextFrame_index}"].update({
                    vid: detections[f"{currentFrame_index}"][vid]
                })
                if printing:
                    print(f"\t{vid}")
            currentFrame_index = nextFrame_index
            continue

        # Build cost matrix for unmatched-current vs new-next IDs
        cost_matrix = np.zeros(
            (len(unmatched_currentFrame_Ids), len(new_nextFrame_mouseIds)),
            dtype=np.float32
        )
        for r, cur_id in enumerate(unmatched_currentFrame_Ids):
            for c, new_id in enumerate(new_nextFrame_mouseIds):
                cost_matrix[r, c] = compute_cost(
                    detections[f"{currentFrame_index}"][cur_id],
                    detections[f"{nextFrame_index}"][new_id],
                    scale_factor,
                    penalty_per_missing,
                    abs_w,
                    abs_h,
                    alpha,
                    epsilon
                )

        # Hungarian assignment
        row_idxs, col_idxs = linear_sum_assignment(cost_matrix)

        # Case 4: apply matches with threshold / stuck logic
        for m in range(len(row_idxs)):
            r_idx = row_idxs[m]
            c_idx = col_idxs[m]
            cur_id = unmatched_currentFrame_Ids[r_idx]
            new_id = new_nextFrame_mouseIds[c_idx]

            if printing:
                print(f"Case 4 {currentFrame_index} - {nextFrame_index}\n\t{cur_id} - {new_id}")

            valid_mouseId_presence[cur_id] = (True, True)

            stuck_condition = (stuck_ids[cur_id][0] and stuck_ids[cur_id][1] < release_id_at_value)
            if (cost_matrix[r_idx, c_idx] <= cost_threshold) or stuck_condition:
                detections[f"{nextFrame_index}"].update({
                    cur_id: detections[f"{nextFrame_index}"].pop(new_id)
                })
                if stuck_condition and printing:
                    print(f"Stucked id {cur_id} released at frame {nextFrame_index} of {vid_name}.")
                stuck_ids[cur_id] = (False, 0)
            else:
                # Replace new_id with the previous frame's cur_id annotation
                detections[f"{nextFrame_index}"].pop(new_id)
                detections[f"{nextFrame_index}"].update({
                    cur_id: detections[f"{currentFrame_index}"][cur_id]
                })
                if printing:
                    print(f"Id {cur_id} stucked at frame {nextFrame_index} of {vid_name}.")
                stuck_ids[cur_id] = (True, stuck_ids[cur_id][1] + 1)

        # Case 5: remaining unmatched current IDs → propagate
        for vid, flags in valid_mouseId_presence.items():
            if (not flags[0]) or flags[1]:
                continue
            valid_mouseId_presence[vid] = (True, True)
            detections[f"{nextFrame_index}"].update({
                vid: detections[f"{currentFrame_index}"][vid]
            })
            if printing:
                print(f"Case 5 {currentFrame_index} - {nextFrame_index}\n\t {vid}")

        # Case 6: leftover new IDs → assign to any free slots
        for new_id in new_nextFrame_mouseIds:
            if new_id not in detections[f"{nextFrame_index}"]:
                continue
            for vid, flags in valid_mouseId_presence.items():
                if flags[0]:
                    continue
                valid_mouseId_presence[vid] = (True, True)
                detections[f"{nextFrame_index}"].update({
                    vid: detections[f"{nextFrame_index}"].pop(new_id)
                })
                if printing:
                    print(f"Case 6 {currentFrame_index} - {nextFrame_index}\n\t{vid} - {new_id}")
                break

        currentFrame_index = nextFrame_index

    return detections


def overlay_annotations_on_video(
    input_video: str,
    annotations: Dict[str, Dict[str, Any]],
    color_bbox: Dict[str, Tuple[int, int, int]],
    color_kpts: Dict[str, Tuple[int, int, int]],
    output_video: str = "output.mp4",
    discard: Tuple[bool, List[int]] = (False, [])
) -> None:
    """
    Description
    -----------
    Draws tracked bounding boxes and keypoints onto a video and writes the
    annotated result to disk. Annotation keys are expected to be stringified
    frame indices ("1", "2", ...).

    Inputs
    ------
    input_video : str
        Path to the source video file.
    annotations : dict
        Per-frame annotations:
        {
          "1": { "id_str": {"bbox": {...}, "keypoints": {...}}, ... },
          "2": { ... },
          ...
        }
    color_bbox : dict
        Mapping from ID string to BGR color, e.g. {"1": (0,0,255), ...}.
    color_kpts : dict
        Mapping from keypoint name to BGR color, e.g. {"nose": (0,255,255), ...}.
    output_video : str, default "output.mp4"
        Path for the output annotated video.
    discard : (bool, list[int]), default (False, [])
        If True and an ID (as int) is in the list, its drawings are skipped.

    Returns
    -------
    None
        Saves the annotated video to `output_video`.
    """

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Could not open video: {input_video}")
        return

    # Retrieve video properties (fallback if FPS is 0/NaN)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fps    = fps if fps and fps > 0 else 30.0

    # Define codec and create writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_index = 1  # adjust if your annotations start at "0"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = f"{frame_index}"
        if frame_key in annotations:
            try:
                for mouse_id, mouse_data in annotations[frame_key].items():
                    # Skip discarded IDs
                    if discard[0]:
                        try:
                            if int(mouse_id) in discard[1]:
                                continue
                        except ValueError:
                            # mouse_id wasn't an int-like string; ignore discard for it
                            pass

                    # Bounding box
                    bbox = mouse_data.get("bbox", {})
                    x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                    x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))

                    # Colors (fallbacks if missing)
                    box_color = color_bbox.get(mouse_id, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Label with ID
                    cv2.putText(
                        frame, f"Mouse {mouse_id}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
                    )

                    # Keypoints
                    keypoints = mouse_data.get("keypoints", {})
                    for kpt_name, kpt_vals in keypoints.items():
                        try:
                            kx, ky, _ = kpt_vals
                            kx, ky = int(kx), int(ky)
                        except Exception:
                            continue  # skip malformed kpt

                        kpt_color = color_kpts.get(kpt_name, (255, 255, 255))
                        cv2.circle(frame, (kx, ky), 4, kpt_color, -1)
                        cv2.putText(
                            frame, kpt_name, (kx + 5, ky),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, kpt_color, 1
                        )
            except Exception as error:
                print(f"Error drawing annotations for frame {frame_index}: {error}")

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print("Finished writing annotated video:", output_video)