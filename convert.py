import onnxruntime as ort
import numpy as np
import torch
import transnetv2

input_video = torch.zeros(1, 3, 100, 27, 48, dtype=torch.uint8) * 255  # .cuda()
input_video = input_video.cuda()
# dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
model = transnetv2.models.trans_net(pretrained=True).cuda()
# state_dict = torch.load("latest.pth")
# model.load_state_dict(state_dict)
model.eval()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = ["input1"]  # + [ "learned_%d" % i for i in range(16) ]
output_names = ["output1"]

torch.onnx.export(model,
                  input_video,
                  "transnetv2.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

ort_session = ort.InferenceSession("transnetv2.onnx")

outputs = ort_session.run(
    None,
    {"input1": np.random.randn(1, 3, 100, 27, 48).astype(np.uint8) * 255},
)
print(outputs[0])
