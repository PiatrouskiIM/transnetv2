import os.path
import torch
import numpy as np
from . import video
from . import functional

FORMAT_BY_EXT = {"pt": "pytorch", "pth": "pytorch", "onnx": "onnx", "mlprogram": "coreml"}


def inference_format_by_extension(model_path: str) -> str:
    model_path = model_path or "path.pt"
    extension = os.path.splitext(model_path)[-1][1:]
    return FORMAT_BY_EXT[extension]


class Predictor:
    # framework: str = None
    # confidence: float = .25
    # min_scene_duration: float = .33
    # batch_size: int = 1
    def __init__(self,
                 framework: str = None,
                 confidence: float = .25,
                 min_scene_duration: float = .33,
                 batch_size: int = 1):
        self.framework = framework
        self.confidence = confidence
        self.min_scene_duration = min_scene_duration
        self.batch_size = batch_size

    def __call__(self, video_path: str):
        return []

    def export(self, save_path: str):
        raise NotImplementedError

    def aggregate_output(self, predictions, fps):
        min_scene_duration_in_frames = int(np.floor(fps * self.min_scene_duration))
        predictions = functional.non_maximum_suppression(sequence=predictions, radius=min_scene_duration_in_frames)
        frame_nos = np.where(predictions > self.confidence)[0]
        timecodes = frame_nos / fps
        return timecodes


class PytorchPredictor(Predictor):
    def __init__(self, model_path: str = None, confidence=.25, min_scene_duration=.33, device="cuda", batch_size=1):
        super().__init__("pytorch", confidence, min_scene_duration, batch_size)
        from .models import trans_net
        self.device = torch.device(device)
        self.model = trans_net(weights=model_path)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, video_path: str):
        fps = video.read_meta(video_path)["fps"]
        frames = video.read_resized_frames(video_path=video_path, size=(48, 27))
        predictions = []
        for batch in functional.get_sliding_window(frames, batch_size=self.batch_size):
            x = torch.Tensor(batch).to(self.device)
            predictions.extend(self.model(x)[:, 25:75].detach().cpu().numpy())
        return self.aggregate_output(np.concatenate(predictions)[:len(frames)], fps)


class ONNXPredictor(Predictor):
    def __init__(self, model_path: str = None, confidence=.25, min_scene_duration=.33, device="cuda", batch_size=1):
        super().__init__("onnx", confidence, min_scene_duration, batch_size)
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = False
        self.session = onnxruntime.InferenceSession(model_path, sess_options)

    def __call__(self, video_path: str):
        fps = video.read_meta(video_path)["fps"]
        frames = video.read_resized_frames(video_path=video_path, size=(48, 27))
        predictions = []
        for batch in functional.get_sliding_window(frames, batch_size=self.batch_size):
            predictions.extend(self.session.run(output_names=None, input_feed={"input": batch})[0][0, 25:75])
        return self.aggregate_output(np.concatenate(predictions)[:len(frames)], fps)


class CoreMLPredictor(Predictor):
    def __init__(self, model_path: str = None, confidence=.25, min_scene_duration=.33, device="cuda", batch_size=1):
        super().__init__("coreml", confidence, min_scene_duration, batch_size)
        import coremltools
        self.ct_model = coremltools.models.MLModel(self.model_path)

    def __call__(self, video_path: str):
        fps = video.read_meta(video_path)["fps"]
        frames = video.read_resized_frames(video_path=video_path, size=(48, 27))
        predictions = []
        for batch in functional.get_sliding_window(frames, batch_size=self.batch_size):
            predictions.extend(self.ct_model.predict({'input': batch})["output"][0][25:75])
        return self.aggregate_output(np.concatenate(predictions)[:len(frames)], fps)


predictor_factory = {"pytorch": PytorchPredictor, "onnx": ONNXPredictor, "coreml": CoreMLPredictor}


class TRANSNETV2:
    def __init__(self, model_path: str = None, confidence=.25, min_scene_duration=.33):
        self.model_path = model_path
        self.framework = inference_format_by_extension(model_path)
        assert self.framework, f"Unrecognized format of model {model_path}."
        self.predictor = predictor_factory[self.framework](model_path, confidence, min_scene_duration)

    def __call__(self, video_path: str):
        return self.predictor(video_path)

    def export(self, export_format: str = "onnx"):
        assert export_format in ["tensorflow", "pytorch", "onnx", "coreml", "mlprogram"], \
            f"Provided export_format: {export_format} is not supported."
