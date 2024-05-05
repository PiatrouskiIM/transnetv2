import cv2
import numpy as np
from typing import Dict, List
from .time_decorators import timeit


def extract_meta(capture: cv2.VideoCapture) -> Dict:
    return dict(frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                fps=capture.get(cv2.CAP_PROP_FPS),
                size=(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))


def read_meta(video_path: str) -> Dict:
    capture = cv2.VideoCapture(video_path)
    meta = extract_meta(capture)
    capture.release()
    return meta


@timeit(message="Read resized frames")
def read_resized_frames(video_path: str, size=(48, 27)) -> np.ndarray:
    import ffmpeg
    w, h = size
    video_stream, err = ffmpeg.input(video_path).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}"
    ).run(capture_stdout=True, capture_stderr=True)
    return np.frombuffer(video_stream, np.uint8).reshape([-1, h, w, 3])


def count_resized_frames_size_mb(video_path: str, size=(48, 27), dtype=np.uint8) -> int:
    return read_meta(video_path)["frame_count"] * np.prod(size) * 3 * dtype().nbytes / 1_048_576
