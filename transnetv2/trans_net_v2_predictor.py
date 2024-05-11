import torch
from .predictor import Predictor
from .models import trans_net
from .utils import inference_format_by_extension


class TransNetV2Predictor(Predictor):
    def __init__(self, model_path: str = None, confidence=.25, min_scene_duration=.33, device="cpu", batch_size=1):
        super().__init__("pytorch", confidence, min_scene_duration, batch_size)
        self.device = torch.device(device)
        self.model = trans_net(weights=model_path)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict_numpy(self, x):
        x = torch.Tensor(x).to(self.device)
        return self.model(x)[:, 25:75].detach().cpu().numpy()

    def save(self, save_path: str):
        assert inference_format_by_extension(save_path) == self.framework
        torch.save(self.model.state_dict(), save_path)
