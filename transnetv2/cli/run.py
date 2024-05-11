import os
import numpy as np
from transnetv2 import TransNetV2Predictor as TRANSNET
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, default="input.mp4", help="input video path.")
    parser.add_argument('-o', "--output", type=str, default="cuts.txt", help="output video path.")
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument("--batch", default=1, type=int, help="device.")
    parser.add_argument("--thresh", default=.25, type=float, help="")
    parser.add_argument("--duration", default=.33, type=float, help="")
    args = parser.parse_args()

    predictor = TRANSNET(model_path="checkpoints/latest.pth",
                         confidence=args.thresh,
                         min_scene_duration=args.duration,
                         batch_size=args.batch)
    timecodes = predictor(args.input)
    np.set_printoptions(2)
    print(timecodes)

    output_directory_name = os.path.dirname(args.output)
    if output_directory_name != "":
        os.makedirs(output_directory_name, exist_ok=True)
    np.savetxt(args.output, (timecodes * 1000).astype(int), fmt="%i")


if __name__ == "__main__":
    main()

