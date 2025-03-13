# Libraries

import numpy as np
from ffprobe import FFProbe
import os
import json
from scipy.optimize import linear_sum_assignment
import cv2
from tqdm import tqdm

# Helper Functions

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


def get_video_resolution(filename):
    """
    Returns (width, height) for the first video stream found in `filename`.
    """
    metadata = FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            print(dir(stream))
            return (int(stream.width), int(stream.height))
        
    return (None, None)


def save_metadata(output_dir, metadata_filename, metadata):
    metadata_outFilePath = os.path.join(output_dir, metadata_filename)

    with open(metadata_outFilePath, 'w') as f:
        json.dump(metadata, f, indent=4)


def load_metadata(source_dir, metadata_filename):
    metadata_filePath = os.path.join(source_dir, metadata_filename)

    with open(metadata_filePath, 'r') as f:
        return json.load(f)
    

def get_bBox_xyxyc(bBox):

    c_x = 0.5*(bBox['x1'] + bBox['x2'])
    c_y = 0.5*(bBox['y1'] + bBox['y2'])
    
    return (bBox['x1'], bBox['y1'], bBox['x2'], bBox['y2'], c_x, c_y)


def get_keypoints_xyxyc(keypoints, bBox, scale_factor=0.02):
    '''
    input: {
        'nose' : [x, y, visible_flag],
        'earL' : [x, y, visible_flag],
        'earR' : [x, y, visible_flag],
        'tailB' : [x, y, visible_flag]
    }
    output: {
        'nose' : (x1, y1, x2, y2, c_x, c_y),
        'earL' : (x1, y1, x2, y2, c_x, c_y),
        'earR' : (x1, y1, x2, y2, c_x, c_y),
        'tailB' : (x1, y1, x2, y2, c_x, c_y)
    }
    '''
    keypoints_xyxyc = {}

    bBox_w = max(0., bBox['x2'] - bBox['x1'])
    bBox_h = max(0., bBox['y2'] - bBox['y1'])

    keypoint_bBox_w = scale_factor * bBox_w
    keypoint_bBox_h = scale_factor * bBox_h

    for keypoint, coordinates in keypoints.items():
        x1 = coordinates[0] - (keypoint_bBox_w/2)
        y1 = coordinates[1] - (keypoint_bBox_h/2)
        x2 = coordinates[0] + (keypoint_bBox_w/2)
        y2 = coordinates[1] + (keypoint_bBox_h/2)

        keypoints_xyxyc[keypoint] = (
            x1, y1, x2, y2, coordinates[0], coordinates[1]
        )

    return keypoints_xyxyc


def get_iou(b1, b2):
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


def get_center_distance(b1, b2, abs_w, abs_h):
    """
    Euclidean distance between centers of two bounding boxes
    in (x1, y1, x2, y2) format.
    """
    c1_x, c1_y = b1[4], b1[5]
    c2_x, c2_y = b2[4], b2[5]

    dx = abs(c1_x - c2_x)
    dy = abs(c1_y - c2_y)

    dx_norm = dx / abs_w if abs_w != 0 else dx
    dy_norm = dy / abs_h if abs_h != 0 else dy

    norm_distance = (dx_norm**2 + dy_norm**2) ** 0.5
    
    img_diagonal = (abs_w**2 + abs_h**2) ** 0.5
    
    return norm_distance/img_diagonal


def iou_keypoints(track, det):
    keypoints_present = 0

    iou = 0
    for keypoint, coordinates in track.items():
        if not (keypoint in det):
            continue
        
        iou += get_iou(coordinates, det[keypoint])
        keypoints_present += 1

    if keypoints_present == 0:
        return 0

    avg_iou = iou / keypoints_present
    
    return (keypoints_present/4) * avg_iou
        

def center_distance_keypoints(track, det, abs_w, abs_h, penalty_per_missing=10):
    keypoints_present = 0

    center_distance = 0
    for keypoint, coordinates in track.items():
        if not (keypoint in det):
            continue
        
        # get and add the Euclidean distance
        center_distance += get_center_distance(coordinates, det[keypoint], abs_w, abs_h)
        keypoints_present += 1

    missing_keypoints = 4 - keypoints_present
    if keypoints_present > 0:
        avg_distance = center_distance / keypoints_present
    else:
        # If no keypoints detected, return infinite
        return 1

    # Add penalty for missing keypoints
    # total_score = avg_distance + missing_keypoints / penalty_per_missing
    total_score = (avg_distance + missing_keypoints)/ (missing_keypoints+1)
    return total_score


def compute_cost(track, detection, scale_factor, penalty_per_missing, abs_w, abs_h, alpha=0.5, epsilon=1e-6):
    """
     lower cost => better match.
    """
    track_bBox_xyxyc = get_bBox_xyxyc(track['bbox'])
    det_bBox_xyxyc   = get_bBox_xyxyc(detection['bbox'])

    iou_bBox_val   = get_iou(track_bBox_xyxyc, det_bBox_xyxyc)
    cdist_bBox_val = get_center_distance(track_bBox_xyxyc, det_bBox_xyxyc, abs_w, abs_h)


    track_keypoints_xyxyc = get_keypoints_xyxyc(track['keypoints'], track['bbox'], scale_factor)
    det_keypoint_xyxyc   = get_keypoints_xyxyc(detection['keypoints'], detection['bbox'], scale_factor)

    iou_keypoints_val = iou_keypoints(track_keypoints_xyxyc, det_keypoint_xyxyc)
    cdist_keypoints_val = center_distance_keypoints(track_keypoints_xyxyc, det_keypoint_xyxyc,  abs_w, abs_h, penalty_per_missing)


    cdist_cost =  (1-alpha)*((cdist_bBox_val + cdist_keypoints_val)/2)
    iou_cost = alpha * ((iou_bBox_val + iou_keypoints_val)/2)
    

    return (cdist_cost - iou_cost)/(cdist_cost + iou_cost + epsilon)


def intial_tracking(firstFrame_detections):
    tracking = {}
    detection_id = 0
    for detection in firstFrame_detections:
        tracking[f"{detection_id}"] = detection

        detection_id += 1

    return {"1" : tracking}


def track(detections, scale_factor, penalty_per_missing,  abs_w, abs_h, alpha, epsilon, cost_threshold=0.5, testing=False):
    tracked_detections = intial_tracking(detections['1'])
    currentFrame_index = 1
    
    while currentFrame_index < (len(detections) - 1):
        # next frame
        nextFrame_index = currentFrame_index + 1

        # i x j cost matrix where, 
        #  i is the number of detections in current frame
        # j is the number od detections in previous frame
        cost_matrix = np.zeros((len(tracked_detections[f'{currentFrame_index}']), len(detections[f'{nextFrame_index}'])), dtype=np.float32)
        
        # print(len(tracked_detections[f'{currentFrame_index}']), len(detections[f'{nextFrame_index}']))
        
        for tracked_id, tracked_annotation in tracked_detections[f'{currentFrame_index}'].items():
            for detected_annotation_index in range(len(detections[f'{nextFrame_index}'])):
                detected_annotation = detections[f'{nextFrame_index}']
                # print(f"\t {int(tracked_id)} {int(detected_annotation_index)}")
                cost_matrix[int(tracked_id)][int(detected_annotation_index)] = compute_cost(
                    tracked_annotation,
                    detected_annotation[detected_annotation_index],
                    scale_factor,
                    penalty_per_missing,
                    abs_w,
                    abs_h,
                    alpha,
                    epsilon
                )


        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        if testing:
            print(cost_matrix)

            print(row_idx, col_idx)

            break

        all_mouseId_list = [0, 1, 2, 3, 4]
        included_mouseId_list = []

        tracked_detection = {}
        for i in range(len(row_idx)):
            if cost_matrix[row_idx[i]][col_idx[i]] <= cost_threshold:
                tracked_detection[f'{row_idx[i]}'] = detections[f'{nextFrame_index}'][col_idx[i]]
            else:
                tracked_detection[f'{row_idx[i]}'] = tracked_detections[f'{currentFrame_index}'][f'{row_idx[i]}']

            included_mouseId_list.append(row_idx[i])

        
        for mouse_id in all_mouseId_list:
            if mouse_id in included_mouseId_list:
                continue
            
            tracked_detection[f'{mouse_id}'] = tracked_detections[f'{currentFrame_index}'][f'{mouse_id}']

        tracked_detections[f'{nextFrame_index}'] = tracked_detection

        # skip to next frame
        currentFrame_index += 1


    return tracked_detections


def overlay_annotations_on_video(input_video, annotations, color_box, color_kpt, output_video="output.mp4", discard=(False, [])):

    cap = cv2.VideoCapture(input_video)

    # Retrieve video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter to save the output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'/'avc1' etc.
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_index = 1  # or 0, depending on how your annotations are keyed
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames in video
        
        if f"{frame_index}" in annotations:
            # Get all mice info for this frame
            for mouse_id, mouse_data in annotations[f"{frame_index}"].items():
                
                if discard[0] and (int(mouse_id) in discard[1]):
                    continue

                # Extract bounding box
                bbox = mouse_data['bbox']
                x1, y1 = int(bbox['x1']), int(bbox['y1'])
                x2, y2 = int(bbox['x2']), int(bbox['y2'])

                # Draw the bounding box
                # color_box = (0, 255, 255)  # e.g. yellow
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box[int(mouse_id)], 2)

                # (Optional) Label the mouse ID
                cv2.putText(frame, f"Mouse {mouse_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box[int(mouse_id)], 2)

                # Draw each keypoint
                keypoints = mouse_data['keypoints']
                for kpt_name, (kx, ky, conf) in keypoints.items():
                    # conf is a confidence score you can use if needed
                    kx, ky = int(kx), int(ky)
                    # color_kpt = (0, 255, 0)  # e.g. green
                    cv2.circle(frame, (kx, ky), 4, color_kpt[kpt_name], -1)

                    # (Optional) label the keypoint name
                    cv2.putText(frame, kpt_name, (kx+5, ky),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_kpt[kpt_name], 1)

        # Write the modified frame to output video
        out.write(frame)

        frame_index += 1

    cap.release()
    out.release()
    print("Finished writing annotated video:", output_video)


# Main

def main():
    cycle = 9
    predictedAnnotated_vids_dir = f"/home/jalal/projects/data/neurocig/vids/results/cycle_{cycle}/annotated"
    vids_predictionOn_path = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/vids/processed/"
    output_dir = "/home/jalal/projects/data/neurocig/vids/results/cycle_{cycle}/tracked"

    os.makedirs(output_dir, exist_ok=True)

    for vid_name in tqdm(os.listdir(vids_predictionOn_path)):
        if not vid_name.endswith('.mp4'):
            continue
        
        orig_vidPath = os.path.join(vids_predictionOn_path, vid_name)

        predicted_labelsPath = os.path.join(predictedAnnotated_vids_dir, f'{vid_name}/labels')

        output_tracked_vidPath = os.path.join(output_dir, vid_name.removesuffix('.mp4'))

        img_w, img_h = get_video_resolution(orig_vidPath)


        # dict { frame_idx: [ { 'bbox':(x,y,w,h), 'keypoints':... }, ... ] }
        detections = {}

        mAnnotated_flag = False
        visiblePercentage = 0.90
        for predicted_label in os.listdir(predicted_labelsPath):
            if predicted_label.endswith('.txt'):
                txt_path = os.path.join(predicted_labelsPath, predicted_label)

                temp_holder = predicted_label.split('_')
                frame_index = int(temp_holder[1].split('.')[0])

                detection = yolo_txt_to_annotation_json(
                    txt_path,
                    frame_index,
                    img_w,
                    img_h,
                    mAnnotated_flag,
                    visiblePercentage,
                    ["nose", "earL", "earR", "tailB"]
                )

        detections.update(detection)

        # sort based on frame index
        detections = dict(sorted(detections.items()))

        # perform tracking
        scale_factor = 0.15
        penalty_per_missing = 100
        alpha = 0.75
        epsilon = 1e-6
        cost_threshold = -0.9
        testing = False
        tracked_detections = track(detections,
            scale_factor,
            penalty_per_missing,
            img_w, img_h, alpha,
            epsilon,
            cost_threshold,
            testing
        )

        save_metadata(output_tracked_vidPath, 'tracked_annotations.json', tracked_detections)

        FinalVideo_path = os.path.join(output_tracked_vidPath, vid_name)

        color_box = {
            0 : (0, 255, 255),
            1 : (0, 255, 128),
            2 : (153, 51, 155),
            3 : (255, 255, 0),
            4 : (255, 0, 255)
        }

        color_kpt = {
            'nose' : (153, 204, 255),
            'earL' : (255, 182, 78),
            'earR' : (255, 102, 102),
            'tailB' : (255, 153, 204)
        }

        discard = (False, [])


        overlay_annotations_on_video(orig_vidPath,
            tracked_detections,
            color_box,
            color_kpt,
            FinalVideo_path,
            discard
        )


if __name__ == '__main__':
    main()
