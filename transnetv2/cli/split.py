import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", type=str, required=True, help="input video path.")
parser.add_argument('-o', "--output", type=str, default="cuts.txt", help="output video path.")
args = parser.parse_args()


timecodes = np.loadtxt(args.output)

SAVE_FOLDER = os.path.splitext(args.input)[0]

os.makedirs(SAVE_FOLDER, exist_ok=True)
# os.system("""ffmpeg -i input_video.mp4 -ss [start_time] -t [duration] -c:v copy -c:a copy output_video.mp4""")

timecodes = np.array([0] + list(timecodes))
timecodes, durations = timecodes[:-1], timecodes[1:] - timecodes[:-1]

list_of_ms2time = lambda y: list(map(lambda x: str(x).split("T")[-1],  pd.to_datetime(y, unit='ms').values))

timecodes, durations = list_of_ms2time(timecodes), list_of_ms2time(durations)

for i, (t, d) in tqdm(enumerate(zip(timecodes, durations))):
    index = str(i).zfill(6)
    os.system(f"""ffmpeg -i '{args.input}' -ss {t} -t {d} -c:v copy -c:a copy '{os.path.join(SAVE_FOLDER, index)}.mp4'""")
