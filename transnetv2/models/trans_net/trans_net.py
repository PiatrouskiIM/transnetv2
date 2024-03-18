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

        self.feat_fusion = nn.Sequential(nn.Linear(in_features=similarity_dim, out_features=128),
                                         nn.ReLU())
        self.hist_fusion = nn.Sequential(nn.Linear(in_features=similarity_dim, out_features=128),
                                         nn.ReLU())
        self.outro = nn.Sequential(nn.Linear(in_features=4864, out_features=1024),
                                   nn.ReLU(),
                                   nn.Dropout(dropout_rate))
        self.head = nn.Linear(1024, 1)

    def forward(self, inputs):
        x = inputs
        x = x.float()
        x = x.div_(255.)

        features = []
        for module in self.levels:
            x = module(x)
            features.append(torch.mean(x, dim=[3, 4]))  # b x c x t x h x w -> b x c x t

        c = x.permute(0, 2, 3, 4, 1).flatten(start_dim=2)  # b x c x t x h x w -> b x t x h x w x c -> b x t x (-1)

        x = torch.cat(features, dim=1).transpose(1, 2)
        x = self.feature_projection(x)
        x = functional.normalize(x, p=2, dim=-1)
        x = functional.limited_pairwise_similarities(x, kernel_size=torch.tensor(self.similarity_dim))
        b = self.feat_fusion(x)

        x = torch.flatten(inputs.transpose(1, 2), start_dim=0, end_dim=1)
        x = functional.get_color_histograms(x).view(inputs.size(0), inputs.size(2), 512)
        x = functional.normalize(x, p=2, dim=-1)
        x = functional.limited_pairwise_similarities(x, kernel_size=torch.tensor(self.similarity_dim))
        a = self.hist_fusion(x)

        x = torch.cat([a, b, c], dim=-1)
        x = self.outro(x)
        x = self.head(x)
        return torch.squeeze(x, dim=-1)


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


def trans_net():
    return TransNetV2()
