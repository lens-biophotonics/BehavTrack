# Libraries

import os
import shutil
import json
from sklearn.model_selection import train_test_split
from collections import Counter

# Helper Functions

def load_metadata(source_dir, metadata_filename):
    metadata_filePath = os.path.join(source_dir, metadata_filename)

    with open(metadata_filePath, 'r') as f:
        return json.load(f)
    

def performSplit(metadata, split_ratio):
    if split_ratio > 1.0:
        split_ratio = 1.0
        
    clusters = [entry['cluster'] for entry in metadata]
    
    activeLearning_data, prediction_data = train_test_split(
        metadata,
        test_size=split_ratio,
        random_state=42,
        stratify=clusters  # stratify 
        )
    
    return activeLearning_data, prediction_data


def getCluster_ratio(data):
    # Count the number of frames per cluster
    cluster_counts = Counter(entry['cluster'] for entry in data)
    # Calculate total number of frames
    total_frames = len(data)
    # Print cluster percentages
    for cluster_id, count in cluster_counts.items():
        percentage = (count / total_frames) * 100
        print(f"Cluster {cluster_id}: {count} frames ({percentage:.2f}%)")


def printDetails(metadata, split_data):
    print("#############")
    print(f"Total frames: {len(metadata)}")
    getCluster_ratio(metadata)
    print("\t#############")
    for split in split_data:
        percentage = (len(split[1][0])/len(metadata))*100
        print(f"{split[0][1]} split: {len(split[1][0])} ({percentage:.2f}%)")
        getCluster_ratio(split[1][0])
        print("\t#############")


def save_metadata(output_dir, metadata_filename, metadata):
    metadata_outFilePath = os.path.join(output_dir, metadata_filename)

    with open(metadata_outFilePath, 'w') as f:
        json.dump(metadata, f, indent=4)


def copyframes(split):
    # destination path for the split frame
    out_path = os.path.join(split[2][1], split[0][0])
    os.makedirs(out_path, exist_ok=True)
    # save the metadata at the output
    save_metadata(split[2][1], split[0][1], split[1][0])

    if split[1][1]:
        for data in split[1][0]:
            frame_name = os.path.basename(data['image_path'])
            # src frame path
            src_framePath = os.path.join(split[2][0], frame_name)
            out_framePath = os.path.join(out_path, frame_name)
            # copy
            shutil.copy2(src_framePath, out_framePath)

    print(f"Copy complete for the split {split[0][1]} with images - {split[1][1]}")


# Main

def main():
    # dirs' path
    source_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/frames"
    output_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/stratifySplit_frames"
    activeLearning_dir = "/mnt/c/Users/karti/chest/CNR/projects/data/neurocig/stratifySplit_frames/activeLearning"
    metadata_filename_al = "activeLearning.json"


    # number of initial frames of manuel annotation
    intialN_annotatonData = 100

    # load the active learning frames metadata
    activeLearning_metadata = load_metadata(output_dir, metadata_filename_al)

    # calculate the split ratio respective to intialN_annotatonData
    split_ratio_aL = round(((intialN_annotatonData/len(activeLearning_metadata))*100)/100, 3)

    # annotation split
    activeLearning_metadata_new, annotation_metadata = performSplit(activeLearning_metadata, split_ratio_aL)

    split_data_aL = (
            ('activeLearning', 'activeLearning.json'),
            (activeLearning_metadata_new, False),
            (source_dir, output_dir)
        )

    split_data_annotations =   (
            ('annotations', 'annotations_metadata.json'),
            (annotation_metadata, True),
            (source_dir, activeLearning_dir)
        )

    printDetails(activeLearning_metadata, [split_data_aL, split_data_annotations])


    prediction_percentage = 30
    nOf_annotations = float((prediction_percentage*len(annotation_metadata))/100)
    split_ratio_predict = round(((nOf_annotations/len(activeLearning_metadata_new))*100)/100, 3)

    # annotation split
    activeLearning_metadata_updated, predict_metadata = performSplit(activeLearning_metadata_new, split_ratio_predict)

    split_data_aL = (
            ('activeLearning', 'activeLearning.json'),
            (activeLearning_metadata_updated, False),
            (source_dir, output_dir)
        )

    split_data_predict =   (
            ('predict', 'predict.json'),
            (predict_metadata, True),
            (source_dir, activeLearning_dir)
        )

    printDetails(activeLearning_metadata_new, [split_data_aL, split_data_predict])


    split_ratio_annotation = 0.15

    # train val split
    train_metadata, val_metadata = performSplit(annotation_metadata, split_ratio_annotation)

    split_data_train = (
            ('train', 'train.json'),
            (train_metadata, False),
            (source_dir, activeLearning_dir)
        )

    split_data_val =   (
            ('val', 'val.json'),
            (val_metadata, False),
            (source_dir, activeLearning_dir)
        )

    printDetails(annotation_metadata, [split_data_train, split_data_val])


    split_data_is = [split_data_aL, split_data_annotations, split_data_train, split_data_val, split_data_predict]

    for split in split_data_is:
        copyframes(split)


if __name__ == "__main__":
    main()