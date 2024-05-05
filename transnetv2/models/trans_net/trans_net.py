import torch
import torch.nn as nn
from .basic_block import BasicBlock
import transnetv2.functional as functional

from torchvision.models._api import WeightsEnum, Weights


class TransNetV2(nn.Module):
    def __init__(self, multiplier=16, n_blocks=3, similarity_dim=101, dropout_rate=.5):
        super(TransNetV2, self).__init__()
        self.similarity_dim = similarity_dim
        block_sizes = [multiplier * (2 ** i) for i in range(n_blocks)]
        self.levels = nn.ModuleList([
            nn.Sequential(BasicBlock(in_channels=3 if block_i == 0 else filters * 2, multiplier=filters),
                          nn.AvgPool3d(kernel_size=(1, 2, 2)))
            for block_i, filters in enumerate(block_sizes)])
        self.feature_projection = nn.Linear(in_features=sum(block_sizes) * 4, out_features=128)
        self.feat_fusion = nn.Sequential(nn.Linear(in_features=similarity_dim, out_features=128), nn.ReLU(inplace=True))
        self.hist_fusion = nn.Sequential(nn.Linear(in_features=similarity_dim, out_features=128), nn.ReLU(inplace=True))
        in_features_outro = (multiplier * 2 ** (n_blocks - 1) * 4) * 3 * 6 + 256
        self.outro = nn.Sequential(nn.Linear(in_features=in_features_outro, out_features=1024), nn.ReLU(inplace=True))
        self.head = nn.Linear(in_features=1024, out_features=1)

    def forward(self, inputs):
        x = inputs
        x = x.float()
        x = x.div_(255.)

        features = []
        for module in self.levels:
            x = module(x)
            features.append(torch.mean(x, dim=[3, 4]))
        c = x.permute(0, 2, 3, 4, 1).flatten(start_dim=2)

        x = torch.cat(features, dim=1).transpose(1, 2)
        x = self.feature_projection(x)
        x = functional.normalize(x, dim=-1)
        x = functional.limited_pairwise_similarities(x, kernel_size=torch.tensor(self.similarity_dim))
        b = self.feat_fusion(x)

        x = torch.flatten(inputs.transpose(1, 2), start_dim=0, end_dim=1)
        x = functional.get_color_histograms(x).view(inputs.size(0), inputs.size(2), 512)
        x = functional.normalize(x, dim=-1)
        x = functional.limited_pairwise_similarities(x, kernel_size=torch.tensor(self.similarity_dim))
        a = self.hist_fusion(x)

        x = torch.cat([a, b, c], dim=-1)
        x = self.outro(x)
        x = self.head(x)
        x = torch.squeeze(x, dim=-1)
        return torch.sigmoid(x)

    def forward_ane(self, inputs):
        inputs = inputs  # .permute(0, 4, 1, 2, 3)
        x = inputs
        x = x.float()
        x = x.div_(255.)

        features = []
        for module in self.levels:
            x = module[0].forward_ane(x)
            x = module[1](x)
            features.append(torch.mean(x, dim=[3, 4]))
        c = x.permute(0, 2, 3, 4, 1).flatten(start_dim=2)

        x = torch.cat(features, dim=1).transpose(1, 2)
        x = self.feature_projection(x)
        x = functional.normalize(x, dim=-1)
        x = functional.limited_pairwise_similarities(x, kernel_size=torch.tensor(self.similarity_dim))
        b = self.feat_fusion(x)

        x = torch.flatten(inputs.transpose(1, 2), start_dim=0, end_dim=1)
        x = functional.get_color_histograms(x).view(inputs.size(0), inputs.size(2), 512)
        x = functional.normalize(x, dim=-1)
        x = functional.limited_pairwise_similarities(x, kernel_size=torch.tensor(self.similarity_dim))
        a = self.hist_fusion(x)

        x = torch.cat([a, b, c], dim=-1)
        x = self.outro(x)
        x = self.head(x)
        x = torch.squeeze(x, dim=-1)
        return torch.sigmoid(x)


class TransNetV2ANE(TransNetV2):
    def forward(self, inputs):
        return self.forward_ane(inputs)


class TransNetV2_Weights(WeightsEnum):
    ClipShots_V1 = Weights(
        url="",
        transforms=lambda x: x,  # partial(ImageClassification, crop_size=224),
        meta={
            # **_COMMON_META,
            # "num_params": 2542856,
            "recipe": None,
            "_docs": """
                These weights improve upon the results of the original paper by using a simple training recipe.
            """,
        },
    )
    DEFAULT = ClipShots_V1

    @staticmethod
    def transforms():
        return lambda x: x

    @staticmethod
    def postprocessing():
        return lambda x: x


def trans_net(*, weights=None, progress=True, **kwargs):
    model = TransNetV2(**kwargs)
    if weights is not None:
        model.load_state_dict(torch.load(weights))#.get_state_dict(progress=progress))
    return model
