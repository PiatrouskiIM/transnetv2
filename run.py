

if __name__ == "__main__":
    import os
    import numpy as np

    import torch
    from tqdm import tqdm
    from transnetv2.models import trans_net, TransNetV2_Weights
    from transnetv2.functional import non_maximum_suppression
    from transnetv2.functional import get_sliding_window
    from transnetv2.video import read_meta, read_resized_frames
    from transnetv2 import TRANSNETV2
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, default="input.mp4", help="input video path.")
    parser.add_argument('-o', "--output", type=str, default="cuts.txt", help="output video path.")
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument("--batch", default=1, type=int, help="device.")
    parser.add_argument("--thresh", default=.25, type=float, help="")
    parser.add_argument('--trace', action="store_true")
    parser.add_argument("--use_nn", action="store_true")
    parser.add_argument("--ms", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = TRANSNETV2("/home/ivan/Projects/transnetv2/latest.pth")

    out = detector(args.input)
    np.set_printoptions(2)
    print(out)
    model = trans_net(weights=None)#TransNetV2_Weights.DEFAULT)
    # state_dict = torch.load("latest.pth")
    state_dict = torch.load("/home/ivan/Projects/transnetv2/latest.pth")

    print(model.load_state_dict(state_dict))
    model.eval()
    model.to(device)

    # import onnxruntime as ort
    # sess_options = ort.SessionOptions()
    # # sess_options.enable_profiling = True
    # ort_session = ort.InferenceSession("../taqtile/shot-on-gpu/checkpoints/transnetv2.onnx")#, sess_options)
    #
    # if args.trace:
    #     input_video = (torch.zeros(args.batch, 3, 100, 27, 48) * 255).to(torch.uint8).to(device)
    #     traced_model = torch.jit.trace(model, input_video)
    #     model = traced_model
    #     model.to(device)
    #     model.eval()

    meta = read_meta(args.input)
    frames = read_resized_frames(video_path=args.input, size=(48, 27))
    predictions = []
    for batch in get_sliding_window(frames, stride=50, kernel_size=100, batch_size=args.batch):
        x = torch.Tensor(batch).to(device)
        predictions.extend(model(x)[:, 25:75].detach().cpu().numpy())
    # pad_left, pad_right = calculate_pad(len(frames))
    # frames = np.concatenate([frames[:1]] * pad_left + [frames] + [frames[-1:]] * pad_right, axis=0)
    # number_frames = len(frames)

    # predictions = []
    # x = frames[None].transpose(0, 4, 1, 2, 3)
    # for i in tqdm(range(0, number_frames - 50 * (args.batch + 1), 50 * args.batch)):
    #     index = (np.arange(args.batch) * 50 + i)[:, None] + np.arange(100)[None]
    #     x = torch.Tensor(frames[index].transpose(0, 4, 1, 2, 3)).to(device)
    #     y = torch.sigmoid(model(x)[:, 25:75]).detach().cpu().numpy()
    #     predictions.extend(y)

        # out = ort_session.run(None,{"input1": x[:, :, i:i + 100]})[0]
        # predictions.append(np_sigmoid(out[0, 25:75]))
    #
    predictions = np.concatenate(predictions)
    predictions = non_maximum_suppression(sequence=predictions, radius=int(np.floor(meta["fps"]) / 3))
    frame_nos = np.where(predictions > args.thresh)[0]
    timecodes = frame_nos / meta["fps"]
    np.set_printoptions(2)
    print(timecodes)
    #
    output_directory_name = os.path.dirname(args.output)
    if output_directory_name != "":
        os.makedirs(output_directory_name, exist_ok=True)
    np.savetxt(args.output, (timecodes * 1000).astype(int), fmt="%i")
