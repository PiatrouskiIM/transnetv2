import os
import numpy as np

from . import video
from . import utils


class Predictor:
    def __init__(self,
                 framework: str = None,
                 confidence: float = .25,
                 min_scene_duration: float = .33,
                 batch_size: int = 1):
        self.framework = framework
        self.confidence = confidence
        self.min_scene_duration = min_scene_duration
        self.batch_size = batch_size

    def predict_numpy(self, x):
        raise NotImplementedError

    def __call__(self, video_path: str):
        fps = video.read_meta(video_path)["fps"]
        min_scene_duration_in_frames = int(np.floor(fps * self.min_scene_duration))
        frames = video.read_resized_frames(video_path=video_path, size=(48, 27))
        predictions = []
        for batch in utils.get_sliding_window(frames, batch_size=self.batch_size):
            predictions.extend(self.predict_numpy(batch).reshape(-1))
        predictions = predictions[:len(frames)]
        if min_scene_duration_in_frames != 0:
            predictions = utils.non_maximum_suppression(sequence=predictions, radius=min_scene_duration_in_frames)
        if self.confidence == 0.0:
            return predictions
        frame_nos = np.where(predictions > self.confidence)[0]
        timecodes = frame_nos / fps
        return timecodes
