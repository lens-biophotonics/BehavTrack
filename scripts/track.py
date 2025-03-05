import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ===================== CONFIGURATION / HYPERPARAMETERS =======================

# Maximum allowable distance or mismatch for a valid match.
# Typically, you'd define a function that returns a "cost" or "distance",
# and if cost > COST_THRESHOLD, you treat it as an invalid match.
COST_THRESHOLD = 0.7

# If a track is unmatched for this many consecutive frames, we delete it.
MAX_UNMATCHED_FRAMES = 15

# If the detection and track have a cost above this threshold, set the cost to large.
LARGE_COST = 1e9

# Next ID for new tracks
TRACK_ID_START = 1

# =============================================================================


def iou_bbox(b1, b2):
    """
    Compute IoU of two bounding boxes in (x1, y1, x2, y2) format.
      b1, b2 = (x1, y1, x2, y2) in the same coordinate system.
    """
    # Intersection
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    iw = max(0., ix2 - ix1)
    ih = max(0., iy2 - iy1)
    inter = iw * ih

    # Union
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union < 1e-9:
        return 0.
    return inter / union


def center_distance(b1, b2):
    """
    Euclidean distance between centers of two bounding boxes
    in (x1, y1, x2, y2) format.
    """
    cx1 = 0.5*(b1[0] + b1[2])
    cy1 = 0.5*(b1[1] + b1[3])
    cx2 = 0.5*(b2[0] + b2[2])
    cy2 = 0.5*(b2[1] + b2[3])
    return np.hypot(cx1 - cx2, cy1 - cy2)


def xywh_to_xyxy(b):
    """
    Convert bounding box from (x_center, y_center, w, h) normalized or pixel
    to corner format (x1, y1, x2, y2). You can adapt for your coordinate system.
    """
    x_c, y_c, w, h = b
    x1 = x_c - w/2
    y1 = y_c - h/2
    x2 = x_c + w/2
    y2 = y_c + h/2
    return (x1, y1, x2, y2)


class MouseKalmanFilter:
    """
    A KalmanFilter wrapper for bounding box [x_center, y_center, width, height].
    You can refine this for better motion modeling.
    """
    def __init__(self, init_bbox, init_frame=0):
        # init_bbox: (x_center, y_center, w, h)
        # We'll track [x, y, s, r] with s ~ scale, r ~ aspect ratio (some standard approach).
        # Or you can do a simpler approach [x, y, w, h] directly.
        # This example is loosely adapted from e.g. SORT/AB3DMOT style filters.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State x = [x, y, s, r, vx, vy, vs]
        # z = [x, y, s, r]
        dt = 1.
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0],
            [0, 1, 0, 0, 0,  dt, 0],
            [0, 0, 1, 0, 0,  0,  dt],
            [0, 0, 0, 1, 0,  0,  0 ],
            [0, 0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0, 0,  0,  1 ]
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=float)
        
        # Process noise and measurement noise are hyperparameters
        self.kf.P[4:,4:] *= 1000.  # Large initial uncertainty for velocities
        self.kf.P *= 10.
        self.kf.R[2:,2:] *= 10.  # Scale, ratio measurement noise

        # Initialize
        x, y, w, h = init_bbox
        s = w*h  # scale ~ area
        r = w/float(h+1e-6)  # aspect ratio
        self.kf.x[:4] = np.array([x, y, s, r]).reshape(-1,1)

        self.update(init_bbox)

    def predict(self):
        self.kf.predict()
        return self.get_bbox()

    def update(self, bbox):
        # bbox is (x, y, w, h)
        # Convert to [x, y, s, r]
        x, y, w, h = bbox
        s = w*h
        r = w/float(h+1e-6)
        z = np.array([x, y, s, r])
        self.kf.update(z)
        return self.get_bbox()

    def get_bbox(self):
        """
        Convert [x, y, s, r] in self.kf.x to (x, y, w, h).
        """
        x, y, s, r = self.kf.x[:4].reshape(-1)
        w = np.sqrt(s*r)
        h = np.sqrt(s/r)
        return (x, y, w, h)


class Track:
    """
    Represents a tracked mouse. Stores KalmanFilter, ID, keypoints, etc.
    """
    def __init__(self, detection, track_id):
        self.id = track_id
        # detection['bbox'] is assumed in (x_center, y_center, w, h)
        self.kf = MouseKalmanFilter(detection['bbox'])
        self.keypoints = detection.get('keypoints', None)  # store if you want
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        predicted_bbox = self.kf.predict()
        self.time_since_update += 1
        return predicted_bbox

    def update(self, detection):
        # detection['bbox'] is (x_center, y_center, w, h)
        self.kf.update(detection['bbox'])
        self.keypoints = detection.get('keypoints', None)
        self.time_since_update = 0
        self.hits += 1

    def get_bbox_xyxy(self):
        """
        Return bounding box in (x1, y1, x2, y2) for use in cost calculations or display.
        """
        x, y, w, h = self.kf.get_bbox()
        return xywh_to_xyxy((x, y, w, h))

    def get_bbox_xywh(self):
        return self.kf.get_bbox()


def compute_cost(track: Track, detection: dict, alpha=0.5):
    """
    Example cost function:
      cost = alpha * (1 - iou) + (1-alpha) * center_distance
    where 0 <= cost < 2. Lower cost => better match.
    
    You could incorporate keypoints, color histograms, etc. 
    """
    track_xyxy = track.get_bbox_xyxy()
    det_xyxy   = xywh_to_xyxy(detection['bbox'])

    iou_val   = iou_bbox(track_xyxy, det_xyxy)
    cdist_val = center_distance(track_xyxy, det_xyxy)

    # Weighted combination
    cost = alpha * (1.0 - iou_val) + (1-alpha) * (cdist_val / 100.0)
    # The center distance is scaled by e.g. 100 to keep it in a smaller range.

    return cost


def associate_detections_to_tracks(tracks, detections, alpha=0.5):
    """
    Build a cost matrix of shape (len(tracks), len(detections)) 
    using compute_cost, then solve the assignment using Hungarian.
    
    Returns:
      matched_pairs: list of (track_idx, detection_idx)
      unmatched_tracks: set of track indices
      unmatched_detections: set of detection indices
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], set(range(len(tracks))), set(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            c = compute_cost(trk, det, alpha=alpha)
            # If the cost is too large, we can clamp it or ignore it.
            if c > COST_THRESHOLD:  
                cost_matrix[i, j] = LARGE_COST
            else:
                cost_matrix[i, j] = c

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matched_pairs = []
    for r, c in zip(row_idx, col_idx):
        # If cost is large, treat as unmatched
        if cost_matrix[r, c] >= LARGE_COST:
            continue
        matched_pairs.append((r, c))

    matched_track_indices = set([m[0] for m in matched_pairs])
    matched_det_indices   = set([m[1] for m in matched_pairs])

    unmatched_tracks = set(range(len(tracks))) - matched_track_indices
    unmatched_detections = set(range(len(detections))) - matched_det_indices
    return matched_pairs, unmatched_tracks, unmatched_detections


def track_multi_mice(all_detections):
    """
    Production-level multi-object tracking logic:
      - For each frame, predict track positions
      - Build cost matrix + Hungarian assignment
      - Update matched tracks, handle unmatched
      - Create new tracks for unmatched detections
      - Remove 'stale' tracks
    Input:
      all_detections: dict { frame_idx: [ { 'bbox':(x,y,w,h), 'keypoints':... }, ... ] }
    Return:
      results_per_frame: dict { frame_idx: { track_id: { 'bbox':..., 'keypoints':... } } }
    """
    frame_indices = sorted(all_detections.keys())
    active_tracks = []
    next_id = TRACK_ID_START

    # We'll store the final bounding boxes for each track per frame.
    results_per_frame = {}

    for frame_idx in frame_indices:
        detections = all_detections[frame_idx]

        # 1) PREDICT
        for trk in active_tracks:
            trk.predict()

        # 2) ASSOCIATE
        matched_pairs, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(active_tracks, detections)

        # 3) UPDATE matched tracks
        for (trk_idx, det_idx) in matched_pairs:
            active_tracks[trk_idx].update(detections[det_idx])

        # 4) For unmatched tracks, just keep them with increased time_since_update
        #    (the track.predict() already incremented time_since_update)

        # 5) CREATE new tracks for unmatched detections
        for ud in unmatched_dets:
            new_trk = Track(detections[ud], next_id)
            active_tracks.append(new_trk)
            next_id += 1

        # 6) REMOVE tracks that have been unmatched too long
        survived_tracks = []
        for trk in active_tracks:
            if trk.time_since_update < MAX_UNMATCHED_FRAMES:
                survived_tracks.append(trk)
        active_tracks = survived_tracks

        # 7) Collect results
        frame_result = {}
        for trk in active_tracks:
            bbox = trk.get_bbox_xywh()  # (x_center, y_center, w, h)
            frame_result[trk.id] = {
                'bbox': bbox,
                'keypoints': trk.keypoints
            }
        results_per_frame[frame_idx] = frame_result

    return results_per_frame

