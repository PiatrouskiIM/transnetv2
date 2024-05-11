import onnxruntime
from ..predictor import Predictor
from ..utils import inference_format_by_extension


class ONNXPredictor(Predictor):
    def __init__(self, model_path: str = None, confidence=.25, min_scene_duration=.33, device="cuda", batch_size=1):
        super().__init__("onnx", confidence, min_scene_duration, batch_size)
        assert inference_format_by_extension(model_path=model_path) == "onnx", "Incorrect model format."
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = False
        self.session = onnxruntime.InferenceSession(model_path, sess_options)

    def predict_numpy(self, x):
        return self.session.run(output_names=None, input_feed={"input": x})[0][0, 25:75]


def save(trans_net_v2, save_path: str, **kwargs) -> None:
    trans_net_v2.onnx(save_path, **kwargs)


load = ONNXPredictor
