from time import time

import ffmpeg
import numpy as np
from transnetv2.models import trans_net
import torch
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str,  # required=True,
                        # default="input.mp4",
                        # default="/media/ubuntu/sdd2t/STEP_AGGREGATION/Wood/input.mp4",
                        default="/media/ubuntu/sdd2t/STEP_AGGREGATION/Video instructions/Как наклеить гидрогелевую пленку на смартфон Видеоинструкция Rock Space!/input.mp4",
                        help="input video path.")
    # parser.add_argument('-o', "--output", type=str, default="output.mp4", help="output video path.")
    # parser.add_argument("--device",  default='cuda', choices=["cuda", "cpu"], help="device.")
    # parser.add_argument("--batch", default=1, type=int, help="device.")
    args = parser.parse_args()

    model = trans_net()
    state_dict = torch.load("latest.pth")
    model.load_state_dict(state_dict)
    model.eval()

    point_time = time()
    print(f"Warmup...")
    input_video = torch.zeros(10, 3, 100, 27, 48, dtype=torch.uint8) * 255
    model = model.cuda()
    input_video = input_video.cuda()
    model(input_video)

    print(f"Warmup...Done in {time() - point_time:.2f} sec.")
    # traced_model = torch.jit.trace(model, input_video)
    # model = traced_model
    # print(f"Warmup...Done in {time() - point_time:.2f} sec.")

    point_time = time()
    print(f"Extracting video frames...")
    video_stream, err = ffmpeg.input(args.input).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
    ).run(capture_stdout=True, capture_stderr=True)

    frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
    print(f"Extracting video frames...Done in {time() - point_time:.2f} sec.")

    pad_left = 25
    pad_right = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74
    x = np.concatenate([frames[:1]] * pad_left + [frames] + [frames[-1:]] * pad_right, 0)

    predictions = []
    for i in tqdm(range(0, len(x) - 100, 50)):
        batch = torch.Tensor(x[i:i + 100][None]).permute(0, 4, 1, 2, 3).cuda()
        single_frame_pred = torch.sigmoid(model(batch))
        predictions.append(single_frame_pred.cpu().detach().numpy()[0, 25:75])
    print("")

    predictions = np.concatenate(predictions)[:len(frames)]
    timecodes = np.where(predictions > .5)[0] / 25
    print(timecodes)
