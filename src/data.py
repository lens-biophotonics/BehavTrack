import sys
from typing import Optional, Tuple
from ffprobe import FFProbe
from fractions import Fraction
import os

def get_video_resolution(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Description
    -----------
    Extracts the resolution (width, height) of the first video stream in a file.

    Inputs
    ------
    filename : str
        Path to a video file (e.g., `.mp4`, `.avi`, `.mov`).

    Returns
    -------
    (width, height) : tuple of (int | None, int | None)
        Width and height in pixels if a video stream is found, 
        otherwise (None, None).
    """

    # Run ffprobe to extract metadata for all streams in the file
    metadata = FFProbe(filename)

    # Look for the first video stream and return its width and height
    for stream in metadata.streams:
        if stream.is_video():
            return int(stream.width), int(stream.height)

    # If no video stream is found, return placeholders
    return None, None



def get_frame_rate(filename: str) -> Optional[int]:
    """
    Description
    -----------
    Extracts the frame rate (frames per second) of the first video stream in a file.
    Handles fractional frame rates (e.g., '30000/1001') and rounds to the nearest integer.

    Inputs
    ------
    filename : str
        Path to a video file (e.g., `.mp4`, `.avi`, `.mov`).

    Returns
    -------
    fps : int or None
        Rounded frame rate in frames per second if a video stream is found,
        otherwise None.
    """

    # Extract metadata for all streams in the file
    metadata = FFProbe(filename)

    # Return frame rate from the first video stream
    for stream in metadata.streams:
        if stream.is_video():
            fr_str = str(stream.framerate)  # e.g., '30', '30000/1001'
            try:
                # Convert fraction string to float, then round
                return round(float(Fraction(fr_str)))
            except (ValueError, ZeroDivisionError):
                return None

    # If no video stream exists, return None
    return None


def get_video_duration(filename: str) -> Optional[float]:
    """
    Description
    -----------
    Extracts the duration (in seconds) of the first video stream in a file.

    Inputs
    ------
    filename : str
        Path to a video file (e.g., `.mp4`, `.avi`, `.mov`).

    Returns
    -------
    duration : float or None
        Duration in seconds if a video stream is found,
        otherwise None.
    """

    # Extract metadata for all streams in the file
    metadata = FFProbe(filename)

    # Return duration from the first video stream
    for stream in metadata.streams:
        if stream.is_video():
            try:
                return float(stream.duration)  # convert to float for seconds
            except (ValueError, TypeError):
                return None

    # If no video stream exists, return None
    return None


def get_min_max_frame_rate(source_dir: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Description
    -----------
    Scans all `.mp4` videos in a directory and returns the minimum 
    and maximum frame rates observed.

    Inputs
    ------
    source_dir : str
        Path to the folder containing video files.

    Returns
    -------
    (min_fps, max_fps) : tuple of int or None
        - min_fps : lowest frame rate among the videos
        - max_fps : highest frame rate among the videos
        If no valid frame rates are found, returns (None, None).
    """

    min_fps, max_fps = None, None

    for video_name in os.listdir(source_dir):
        if video_name.endswith(".mp4"):
            video_path = os.path.join(source_dir, video_name)
            fps = get_frame_rate(video_path)  # assumes get_frame_rate() is defined
            if fps is not None:
                if min_fps is None or fps < min_fps:
                    min_fps = fps
                if max_fps is None or fps > max_fps:
                    max_fps = fps

    return min_fps, max_fps


def get_min_max_resolution(source_dir: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Description
    -----------
    Scans all `.mp4` videos in a directory and returns the minimum 
    and maximum resolutions found.

    Inputs
    ------
    source_dir : str
        Path to the folder containing video files.

    Returns
    -------
    (min_res, max_res) : tuple
        - min_res : (width, height) of the smallest resolution video, or None if no valid videos
        - max_res : (width, height) of the largest resolution video, or None if no valid videos
    """

    min_w, min_h = sys.maxsize, sys.maxsize
    max_w, max_h = -sys.maxsize - 1, -sys.maxsize - 1
    found_any = False

    for video_name in os.listdir(source_dir):
        if video_name.endswith(".mp4"):
            video_path = os.path.join(source_dir, video_name)
            w, h = get_video_resolution(video_path)  # assumes this helper is defined

            if w is not None and h is not None:
                found_any = True
                min_w, min_h = min(min_w, w), min(min_h, h)
                max_w, max_h = max(max_w, w), max(max_h, h)

    if not found_any:
        return None, None

    return (min_w, min_h), (max_w, max_h)


def get_nb_frames(filename: str) -> Optional[int]:
    """
    Description
    -----------
    Returns the number of frames in the first video stream.

    Inputs
    ------
    filename : str
        Path to a video file.

    Returns
    -------
    n_frames : int or None
        Number of frames if available; otherwise None.
    """
    metadata = FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            return int(stream.nb_frames)
    return None