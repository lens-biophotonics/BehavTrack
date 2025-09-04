import os
from typing import Optional, Union
import ffmpeg

def pre_process_video(
    source_dir: str,
    output_dir: str,
    video_file: str,
    overwrite: bool = True,
    duration: Optional[Union[int, float]] = 900,
    fps: int = 15,
    w: Union[int, str] = "iw",
    h: Union[int, str] = "ih",
    use_gpu: bool = True,
) -> Optional[str]:
    """
    Description
    -----------
    Trim a video, apply gamma/brightness/contrast normalization, convert to grayscale,
    optionally scale, and write HEVC output (NVENC when available).

    Inputs
    ------
    source_dir : str
        Directory containing the input video.
    output_dir : str
        Directory to write the processed video (created if missing).
    video_file : str
        File name of the video to process.
    overwrite : bool, default True
        If False and output exists, skip processing.
    duration : int | float | None, default 900
        Number of seconds to keep from the start. Use None to keep full length.
    fps : int, default 15
        Output frame rate.
    w : int | str, default "iw"
        Target width for scaling (e.g., 640) or "iw" to keep input width.
    h : int | str, default "ih"
        Target height for scaling (e.g., 480) or "ih" to keep input height.
    use_gpu : bool, default True
        If True, request CUDA hw acceleration and NVENC codec. Falls back to CPU if unavailable.

    Returns
    -------
    output_path : str | None
        Full path to the processed file if created, otherwise None.
    """
    # Resolve paths
    input_path = os.path.join(source_dir, video_file)
    output_path = os.path.join(output_dir, video_file)

    # Early exits and setup
    if not os.path.exists(input_path):
        print(f"Skipping {video_file}: input file not found at {input_path}.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    if not overwrite and os.path.exists(output_path):
        print(f"Skipping {video_file}: already exists in {output_dir}.")
        return None

    # Build input kwargs
    in_kwargs = {}
    if duration is not None:
        # Trim from the start for 'duration' seconds
        in_kwargs["t"] = float(duration)

    if use_gpu:
        # Request CUDA hwaccel; ffmpeg will ignore if not supported
        in_kwargs["hwaccel"] = "cuda"

    try:
        print(f"\tProcessing {video_file}: gamma/brightness/contrast → grayscale → scale → {fps} fps")

        # Input
        stream = ffmpeg.input(input_path, **in_kwargs)

        # Convert to grayscale first (single channel)
        stream = stream.filter("format", "gray")

        # Normalize exposure/contrast (tuned for mice cage videos)
        stream = stream.filter("eq", gamma=1.8, brightness=0.17, contrast=1.3)

        # Scale using high-quality Lanczos (keep 'iw/ih' if strings were passed)
        stream = stream.filter("scale", w, h, flags="lanczos")

        # Choose codec: NVENC if GPU requested, otherwise libx265 (CPU HEVC)
        vcodec = "hevc_nvenc" if use_gpu else "libx265"

        # Output args
        out = (
            ffmpeg
            .output(stream, output_path, vcodec=vcodec, r=fps)
            .global_args("-hide_banner", "-loglevel", "error")
        )

        # Run (force overwrite irrespective of 'overwrite' flag; we've handled that above)
        out.run(overwrite_output=True)

        print(f"\tProcessing completed -> {output_path}")
        return output_path

    except ffmpeg.Error as e:
        # Decode stderr if available for clearer diagnostics
        try:
            err_text = e.stderr.decode("utf-8", errors="ignore")
        except Exception:
            err_text = str(e)
        print(f"Error processing {video_file}: {err_text}")
        return None

    except Exception as e:
        print(f"Unexpected error processing {video_file}: {e}")
        return None
