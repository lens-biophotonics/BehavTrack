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
    metadata_filename_main = "frames_info.json"

    # load the frames metadata
    metadata_main = load_metadata(source_dir, metadata_filename_main)

    # Initial Split Percentages [active learning - 85, test - 15]
    split_ratio_main = 0.15

    # perform intial split
    activeLearning_data, test_data = performSplit(metadata_main, split_ratio_main)
    split_data_main = [(
            ('activeLearning', 'activeLearning.json'),
            (activeLearning_data, False),
            (source_dir, output_dir)
        ),
        (
            ('test', 'test.json'),
            (test_data, True),
            (source_dir, output_dir)
        )]

    printDetails(metadata_main, split_data_main)

    # copy the frames to their appropriate dirs
    for split in split_data_main:
        copyframes(split)

if __name__ == "__main__":
    main()