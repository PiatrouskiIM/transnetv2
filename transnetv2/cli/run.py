import os
import numpy as np

import torch
from transnetv2 import TransNetV2Predictor
import argparse


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command')
    parser_run = subparsers.add_parser('run', help='Run the TransNetV2 model')

    parser_run.add_argument('-i', "--input", type=str, required=True, help="input video path.")
    parser_run.add_argument('-o', "--output", type=str, default="cuts.txt", help="output video path.")
    parser_run.add_argument('--weights', type=str, default=None)
    parser_run.add_argument("--batch", default=1, type=int, help="device.")
    parser_run.add_argument("--thresh", default=.25, type=float, help="")
    parser_run.add_argument("--duration", default=.33, type=float, help="")
    parser_run.add_argument('-v', "--verbose", action="store_true", help="Increase output verbosity.")
    args = parser.parse_args()

    predictor = TransNetV2Predictor(model_path=args.weights,
                                    confidence=args.thresh,
                                    min_scene_duration=args.duration,
                                    batch_size=args.batch,
                                    device="cuda" if torch.cuda.is_available() else "cpu")
    timecodes = predictor(args.input)
    if args.verbose:
        np.set_printoptions(2)
        print(timecodes)

    output_directory_name = os.path.dirname(args.output)
    if output_directory_name != "":
        os.makedirs(output_directory_name, exist_ok=True)
    np.savetxt(args.output, timecodes)


if __name__ == "__main__":
    main()
