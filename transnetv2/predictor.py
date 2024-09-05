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

    def __call__(self, video_path: str, fps=None):
        frames = video.read_resized_frames(video_path=video_path, size=(48, 27), fps=fps)
        fps = fps or video.read_meta(video_path)["fps"]
        min_scene_duration_in_frames = int(fps * self.min_scene_duration)
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

    def stream(self, video_path: str, fps=None):
        frames = video.read_resized_frames(video_path=video_path, size=(48, 27), fps=fps)
        fps = fps or video.read_meta(video_path)["fps"]
        min_scene_duration_in_frames = int(fps * self.min_scene_duration)
        locations, heights = [], []
        for i, batch in enumerate(utils.get_sliding_window(frames, batch_size=self.batch_size, verbose=False)):
            current_scores = self.predict_numpy(batch).reshape(-1)
            new_locations = np.where(current_scores > self.confidence)[0]
            locations.extend(new_locations + i * self.batch_size * 50)
            heights.extend(current_scores[new_locations])
            while len(locations) > 1:
                i_to_delete = 0
                if locations[1] - locations[0] > min_scene_duration_in_frames:
                    if locations[0] < len(frames):
                        yield locations[0] / fps
                else:
                    i_to_delete += heights[0] > heights[1]
                del locations[i_to_delete], heights[i_to_delete]
        if len(locations) > 0 and locations[0] < len(frames):
            yield locations[0] / fps
