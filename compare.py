import sys

import coremltools as ct
from transnetv2.functional import *
from transnetv2.models import trans_net
from transnetv2.models.trans_net._trans_net import TransNetV2


# from transnetv2.models.trans_net._trans_net import old_to_new


# old_to_new()
model = trans_net()
state_dict = torch.load("latest.pth")
model.load_state_dict(state_dict)
model.eval()
# print(model)
# sys.exit(0)
model_gt = TransNetV2()
state_dict = torch.load("transnetv2-pytorch-weights.pth")
model_gt.load_state_dict(state_dict)
model_gt.eval()


with torch.no_grad():
    input_video = torch.zeros(1, 3, 100, 27, 48, dtype=torch.uint8) * 255  # .cuda()

    x = input_video.permute(0, 2, 3, 4, 1)
    a = model(input_video)
    b = torch.squeeze(model_gt(x)[0], -1)

    print(torch.linalg.norm(a - b))

    # # single_frame_pred, all_frame_pred = model(input_video)
    # # single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
    # # all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

    # traced_model = torch.jit.trace(model, input_video)

    # model_from_torch = ct.convert(traced_model,
    #                               convert_to="mlprogram",
    #                               minimum_deployment_target=ct.target.iOS16,
    #                               inputs=[ct.TensorType(name="input", shape=(1, 3, 100, 27, 48))])
    # model_from_torch.save("transnetv2-reduced.mlpackage")

    # model_from_torch = ct.convert(
    #     traced_model,
    #     minimum_deployment_target=ct.target.iOS16,
    #     convert_to="neuralnetwork",
    #     inputs=[ct.TensorType(name="input", shape=(1, 3, 100, 27, 48))]
    # )
    #
    # model_from_torch.save("transnetv2.mlmodel")
