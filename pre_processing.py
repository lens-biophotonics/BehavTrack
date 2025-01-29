# Libraries

import os
import sys
import shutil
import ffmpeg
from ffprobe import FFProbe



# Helper Functions

def copy_videos(source_dir, destination_dir, overwrite=True):
    """Copy 'pre' eCig and Cig videos from source directory to destination directory."""
    os.makedirs(destination_dir, exist_ok=True)

    # Initialize counters
    cig = 0
    eCig = 0
    aria = 0
    total = 0

    print("Copying eCig, Cig 'pre' and aria videos.")

    # Validate if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Get existing files in the destination directory (for faster lookup)
    existing_files = set(os.listdir(destination_dir))

    # Iterate over files in the source directory
    for video_file in os.listdir(source_dir):
        full_source_path = os.path.join(source_dir, video_file)

        # Count total .mp4 files
        if video_file.endswith(".mp4"):
            total += 1
        else:
            continue

        # Skip if the file already exists in the destination and `overwrite` is False
        if not(overwrite) and (video_file in existing_files):
            continue
        
        
        if "aria" in video_file.lower():
            aria += 1
        elif "pre" in video_file.lower():
            if "ecig" in video_file.lower():
                eCig += 1
            elif "cig" in  video_file.lower():
                cig += 1
        else:
            continue

        # Copy file to the destination directory
        shutil.copy(full_source_path, destination_dir)

    print(f"Copy complete!\n\t eCig pre: {eCig}, Cig pre: {cig} and aria: {aria}. Total: {eCig + cig + aria} out of {total}")


def get_video_resolution(filename):
    """
    Returns (width, height) for the first video stream found in `filename`.
    """
    metadata = FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            return (int(stream.width), int(stream.height))
    return (None, None)


def get_frame_rate(filename):
    metadata = FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            return int(stream.framerate)
        
    return None


def get_video_duration(source_dir, video_file):
    """
    Returns (width, height) for the first video stream found in `filename`.
    """
    filename = os.path.join(source_dir, video_file)

    metadata = FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            return float(stream.duration)
    return None


def process_video(source_dir, output_dir, video_file, overwrite=True, duration=900,  fps=15, w='iw', h='ih'):
    """Trim video, apply uniform gamma correction and brightness adjustment, convert to grayscale."""
    # Define input and output paths
    input_path = os.path.join(source_dir, video_file)
    output_path = os.path.join(output_dir, video_file)

    # Check if file already exists and skip if overwrite is False
    if not overwrite and os.path.exists(output_path):
        print(f"Skipping {video_file}: already exists in {output_dir}.")
        return

    try:
        # Apply gamma correction, brightness, contrast adjustments, and convert to grayscale
        print("\tApplying gamma correction, brightness normalization, and converting to grayscale.")
        (
            ffmpeg
            .input(input_path, t=duration, hwaccel="cuda")  # Trim video to specified duration and hardware acceleration using the gpu
            .filter('format', 'gray')  # convert video to grayscale
            .filter('eq', gamma=1.8, brightness=0.17, contrast=1.3)  # EQ Adjustments
            .filter('scale', w,  h, flags='lanczos')  # Clamp pixel values
            .output(output_path, vcodec="hevc_nvenc", r=fps)  # nvidia vcodec
            .run(overwrite_output=True)
        )
        print(f"\tProcessing completed -> {output_path}")

    except ffmpeg.Error as e:
        print(f"Error processing {video_file}: {e.stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error processing {video_file}: {e}")



# Main()

def main():
    # Copy the right videos from the source directory
    source_dir = "/mnt/olimpo1/neurocig/data"
    destination_dir = "/home/jalal/projects/data/neurocig/vids/raw"

    copy_videos(source_dir, destination_dir, overwrite=False)

    # change the source path
    source_dir = "/home/jalal/projects/data/neurocig/vids/raw/"


    # get the min resolution
    min_w = sys.maxsize
    min_h = sys.maxsize

    max_w = -sys.maxsize - 1
    max_h = -sys.maxsize - 1

    for video_file in os.listdir(source_dir):
        video_path = os.path.join(source_dir, video_file)

        w, h = get_video_resolution(video_path)
        min_w = min(min_w, w)
        min_h = min(min_h, h)

        max_w = max(max_w, w)
        max_h = max(max_h, h)

    print(f"The minimun resolution is {min_w}x{min_h} and the maximum resolution is {max_w}x{max_h}.")

    min_fps = sys.maxsize
    max_fps = -sys.maxsize - 1

    for video_file in os.listdir(source_dir):
        video_path = os.path.join(source_dir, video_file)
        fps = get_frame_rate(video_path)

        min_fps = min(min_fps, fps)
        max_fps = max(max_fps, fps)

    print(f"The maximum fps is {max_fps} and the minimum fps is {min_fps}")

    # process the selected videos
    output_dir = "/home/jalal/projects/data/neurocig/vids/processed/"
    os.makedirs(output_dir, exist_ok=True)

    overwrite = True
    for_infernece = False
    for video_file in os.listdir(source_dir):
        if video_file.endswith('.mp4'):
            duration = 900
            fps = min_fps
            if for_infernece:
                if "aria" in video_file.lower():
                    duration = get_video_duration(source_dir, video_file)
                fps = get_frame_rate(source_dir, video_file)
            process_video(source_dir, output_dir, video_file, overwrite, duration, fps, w=min_w, h=min_h)
            

if __name__ == "__main__":
    main()