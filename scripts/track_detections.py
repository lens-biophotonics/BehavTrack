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
    keypoint_names=None,
    tracked=False
):
    """
    Reads a YOLO-like .txt (with bbox + 4 keypoints in normalized coords),
    and returns a dictionary in the original annotation style:

    {
      "image_filename": [
        "Tracking number": {
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
    if tracked:
        annotations = {image_filename: {}}

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

        annotation = {
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "keypoints": keypoints_dict,
            "mAnnotated": mAnnotated_flag
        }

        if tracked:
            tracking_id = int(tokens[17])
            annotations[image_filename].update({
                f'{tracking_id}' : annotation
            })
        else:
            annotations[image_filename].append(annotation)

    return annotations


def get_video_resolution(filename):
    """
    Returns (width, height) for the first video stream found in `filename`.
    """
    metadata = FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
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


def track(vid_name, detections, scale_factor, penalty_per_missing,  abs_w, abs_h, alpha, epsilon, cost_threshold=0.5, printing=False):
    currentFrame_index = 1

    while currentFrame_index <= (len(detections) - 1):
        # next frame
        nextFrame_index = currentFrame_index + 1

        # check if the next frame index is valid or not
        index_flag = f"{nextFrame_index}" in detections
        index_jump = False
        # until we find a valid next frame
        while not index_flag:
            if nextFrame_index == len(detections):
                break       
            
            print(f"Missing index {nextFrame_index} - {vid_name}")
            nextFrame_index += 1
            index_flag = f"{nextFrame_index}" in detections
            index_jump = True
        # if no valid next frame found
        if not index_flag:
            break
        # if there was a next frame index skip, jump to 
        # that index as the current index
        if index_jump:
            currentFrame_index = nextFrame_index
            continue
        
        # To store if the current Frame have valid mouse Ids 
        # and if they exist in the next frame.
        # mouse_ID : (currentFrame_exist, nextFrame_exist) 
        valid_mouseId_presence = {
            '1' : (False, False),
            '2' : (False, False),
            '3' : (False, False),
            '4' : (False, False),
            '5' : (False, False),
        }
        for currentFrame_mouseId, _ in detections[f'{currentFrame_index}'].items():
            valid_mouseId_presence[currentFrame_mouseId] = (
                True, (currentFrame_mouseId in detections[f'{nextFrame_index}'])
            )

        # To store new mouse Ids appeared in the next frame
        new_nextFrame_mouseIds = []
        for nextFrame_mouseId, _ in detections[f'{nextFrame_index}'].items():
            if nextFrame_mouseId in detections[f'{currentFrame_index}']:
                continue

            new_nextFrame_mouseIds.append(nextFrame_mouseId)

        # flag to know if all the current frame mouse ids
        # are matched to the next frame
        valid_toSkip_flag = True
        # get the number of unmatached mouse ids in the current frame
        unmatched_currentFrame_Ids = []
        for valid_currentFrame_ids, existFlags in valid_mouseId_presence.items():
            if existFlags[1]:
                continue
            
            # when not matched 
            if existFlags[0]:
                unmatched_currentFrame_Ids.append(valid_currentFrame_ids)
                valid_toSkip_flag = False


        # when all the current frame mouse ids and the next frame mouse ids
        # are matched and nothing new exist. Skip to the next frame as the current 
        # frame
        if valid_toSkip_flag and (len(new_nextFrame_mouseIds) == 0):
            if printing:
                print(f"case 1 {currentFrame_index} - {nextFrame_index}")
            currentFrame_index = nextFrame_index
            continue
        # when all the mouse ids from the current frames are matched but there
        # is a new mouse id present in the next frame
        # Note: this way of doing is viable here because we know 'at max' there can be only 
        # 5 mouse ids in any frame
        elif valid_toSkip_flag and (len(new_nextFrame_mouseIds) > 0):
            if printing:
                print(f"Case 2 {currentFrame_index} - {nextFrame_index}:")
            # for the new mouse id
            for new_nextFrame_mouseId in new_nextFrame_mouseIds:
                # check for the mouse id that doesn't present in the current frame
                for valid_mouseId, existFlags in valid_mouseId_presence.items():
                    if existFlags[0]:
                        continue
                    
                    # when found set its valid existence to true
                    valid_mouseId_presence[valid_mouseId] = (True, True)
                    # update the next frame new mouse id to the valid non exist one
                    detections[f'{nextFrame_index}'].update({
                        valid_mouseId : detections[f'{nextFrame_index}'].pop(new_nextFrame_mouseId)
                    })

                    if printing:
                        print(f"\t{valid_mouseId} - {new_nextFrame_mouseId}")
                    # To skip to the next new mouse id in the next frame
                    break
            
            # Skip to the next frame as the current frame
            currentFrame_index = nextFrame_index
            continue
        # when few of the mouse ids from the current frame are not matched but there
        # are also no new mouse ids present in the next frame
        if not(valid_toSkip_flag) and (len(new_nextFrame_mouseIds) == 0):
            if printing:
                print(f"Case 3 {currentFrame_index} - {nextFrame_index}")
            #  check for the mouse id that doesn't have a match in the current frame
            for valid_mouseId, existFlags in valid_mouseId_presence.items():
                if not(existFlags[0]) or existFlags[1]:
                    continue

                # when found set its valid matched existence to true
                valid_mouseId_presence[valid_mouseId] = (True, True)
                # add it to the next frame 
                detections[f'{nextFrame_index}'].update({
                        valid_mouseId : detections[f'{currentFrame_index}'][valid_mouseId]
                    })
                
                if printing:
                    print(f"\t{valid_mouseId}")
                
            # Skip to the next frame as the current frame
            currentFrame_index = nextFrame_index
            continue

        
        # set a i x j cost matrix where, 
        #   i is the number of detections in current frame
        #   j is the number od detections in previous frame
        cost_matrix = np.zeros(
            (len(unmatched_currentFrame_Ids), len(new_nextFrame_mouseIds)),
            dtype=np.float32
        )
        # populate the cost matrix
        for row_index in  range(len(unmatched_currentFrame_Ids)):
            # for every current frame unmatched mouse id 
            unmatched_currentFrame_Id = unmatched_currentFrame_Ids[row_index]

            # we compute its cost with every new mouse id in the next frame
            for col_index in range(len(new_nextFrame_mouseIds)):
                new_nextFrame_mouseId = new_nextFrame_mouseIds[col_index]
                
                cost_matrix[row_index][col_index] = compute_cost(
                    detections[f'{currentFrame_index}'][unmatched_currentFrame_Id],
                    detections[f'{nextFrame_index}'][new_nextFrame_mouseId],
                    scale_factor,
                    penalty_per_missing,
                    abs_w,
                    abs_h,
                    alpha,
                    epsilon
                )

        # Best match set of unmatched current frame and next frame mouse ids
        row_indexs, col_indexs = linear_sum_assignment(cost_matrix)

        # for every match
        for match_index in range(len(row_indexs)):
            # get the matched row and col index
            matched_rowIndex = row_indexs[match_index]
            matched_colIndex = col_indexs[match_index]
            # get the matched id from the corresponding frames
            unmatched_currentFrame_Id = unmatched_currentFrame_Ids[matched_rowIndex]
            new_nextFrame_mouseId = new_nextFrame_mouseIds[matched_colIndex]

            if printing:
                print(f"Case 4 {currentFrame_index} - {nextFrame_index}\n\t{unmatched_currentFrame_Id} - {new_nextFrame_mouseId}")

            # update the valid matched existence of the matched current frame
            # mouse id
            valid_mouseId_presence[unmatched_currentFrame_Id] = (True, True)

            # If the computed cost between the two ids is less or equal to the
            # cost thresold, simply update the new mouse id in the next frame
            # with the current frame matched id 
            if cost_matrix[matched_rowIndex][matched_colIndex] <= cost_threshold:
                detections[f'{nextFrame_index}'].update({
                    unmatched_currentFrame_Id : detections[f'{nextFrame_index}'].pop(new_nextFrame_mouseId)
                })
            else: 
                # otherwise, remove the new mouse id and its data in the next frame and simply replace it with 
                # current frame matched mouse id and its data.
                detections[f'{nextFrame_index}'].pop(new_nextFrame_mouseId)

                detections[f'{nextFrame_index}'].update({
                    unmatched_currentFrame_Id : detections[f'{currentFrame_index}'][unmatched_currentFrame_Id]
                })


        # check for the mouse id that doesn't have a match in the current frame,
        # even after the cost comparision
        for valid_mouseId, existFlags in valid_mouseId_presence.items():
            if not(existFlags[0]) or existFlags[1]:
                continue

            # updates its valid matched existence to true
            valid_mouseId_presence[valid_mouseId] = (True, True)
            # add it to the next frame 
            detections[f'{nextFrame_index}'].update({
                    valid_mouseId : detections[f'{currentFrame_index}'][valid_mouseId]
                })
            
            # Skip to the next frame as the current frame
            if printing:
                print(f"case 5 {currentFrame_index} - {nextFrame_index}\n\t {valid_mouseId}")

        # for every new mouse id we say in the next ffame
        for new_nextFrame_mouseId in new_nextFrame_mouseIds:
            # check there is any new mouse id in the next frame that did not get any match
            # with the cost comparision check
            if not(new_nextFrame_mouseId in detections[f'{nextFrame_index}']):
                continue
            
            # look for the non exist valid mouse id in the current frame
            for valid_mouseId, existFlags in valid_mouseId_presence.items():
                if existFlags[0]:
                    continue
                
                # when found set its valid existence to true
                valid_mouseId_presence[valid_mouseId] = (True, True)
                # update the next frame new mouse id to the valid non exist one
                detections[f'{nextFrame_index}'].update({
                    valid_mouseId : detections[f'{nextFrame_index}'].pop(new_nextFrame_mouseId)
                })

                if printing:
                    print(f"Case 6 {currentFrame_index} - {nextFrame_index}\n\t{valid_mouseId} - {new_nextFrame_mouseId}")
                # To skip to the next new mouse id in the next frame
                break

        # go to next frame
        currentFrame_index = nextFrame_index

    return detections


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
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box[mouse_id], 2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color_box[int(mouse_id)], 2)


                # (Optional) Label the mouse ID
                cv2.putText(frame, f"Mouse {mouse_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box[mouse_id], 2)
                # cv2.putText(frame, f"Mouse {mouse_id}", (x1, y1 - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box[int(mouse_id)], 2)

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
    predictedAnnotated_vids_dir = f""
    vids_predictionOn_path = ""
    output_dir = f""

    os.makedirs(output_dir, exist_ok=True)

    for vid_name in tqdm(os.listdir(vids_predictionOn_path)):
        if not vid_name.endswith('.mp4'):
            continue
        
        orig_vidPath = os.path.join(vids_predictionOn_path, vid_name)

        predicted_labelsPath = os.path.join(predictedAnnotated_vids_dir, f'{vid_name.removesuffix(".mp4")}/labels')

        output_tracked_vidPath = os.path.join(output_dir, vid_name.removesuffix(".mp4"))
        os.makedirs(output_tracked_vidPath, exist_ok=True)

        img_w, img_h = get_video_resolution(orig_vidPath)

        # dict { frame_idx: [ { 'bbox':(x,y,w,h), 'keypoints':... }, ... ] }
        detections = {}

        mAnnotated_flag = False
        visiblePercentage = 1.0
        for predicted_label in os.listdir(predicted_labelsPath):
            if predicted_label.endswith('.txt'):
                txt_path = os.path.join(predicted_labelsPath, predicted_label)
                temp_holder = predicted_label.split('_')
                frame_index = f"{int(temp_holder[len(temp_holder)-1].split('.')[0])}"
                detection = yolo_txt_to_annotation_json(
                    txt_path,
                    frame_index,
                    img_w,
                    img_h,
                    mAnnotated_flag,
                    visiblePercentage,
                    ["nose", "earL", "earR", "tailB"],
                    tracked=True
                )
                detections.update(detection)
        # sort based on frame index
        detections = dict(sorted(detections.items()))
        save_metadata(output_tracked_vidPath, 'detections.json', detections)

        detections = load_metadata(output_tracked_vidPath, f'detections.json')

        # perform tracking
        scale_factor = 0.15
        penalty_per_missing = 100
        alpha = 0.75
        epsilon = 1e-6
        cost_threshold = 0.0
        printing = False
        tracked_detections = track(
            vid_name,
            detections,
            scale_factor,
            penalty_per_missing,
            img_w, img_h, alpha,
            epsilon,
            cost_threshold,
            printing
        )
        save_metadata(output_tracked_vidPath, f'tracked_annotations.json', tracked_detections)

        FinalVideo_path = os.path.join(output_tracked_vidPath, vid_name)
        color_box = {
            '1' : (0, 0, 255),
            '2' : (0, 191, 255),
            '3' : (0, 255, 0),
            '4' : (255, 255, 0),
            '5' : (255, 0, 191)
        }
        color_kpt = {
            'nose' : (0, 255, 255),
            'earL' : (255, 102, 102),
            'earR' : (140, 102, 255),
            'tailB' : (0, 128, 255)
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
