import os
import numpy as np
from tqdm import tqdm

FORMAT_BY_EXT = {"pt": "pytorch", "pth": "pytorch", "onnx": "onnx", "mlprogram": "coreml"}


def inference_format_by_extension(model_path: str) -> str:
    model_path = model_path or "path.pt"
    extension = os.path.splitext(model_path)[-1][1:]
    return FORMAT_BY_EXT[extension]


def non_maximum_suppression(sequence, radius: int = 25):
    sequence = np.array(sequence)
    i = 0
    while i < len(sequence) - radius:
        index = np.argmax(sequence[i:i + radius]) + i
        value = sequence[index]
        sequence[i:i + radius] = 0.
        sequence[index] = value
        i = max(index, i + 1)

    index = np.argmax(sequence[-radius:]) + len(sequence) - radius
    value = sequence[index]
    sequence[i:i + radius] = 0.
    sequence[index] = value
    return sequence


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_sliding_window(inputs, kernel_size=100, stride=50, batch_size=1, padding=1, verbose=True):
    def calculate_pad(n, step_size=stride):
        return int(step_size / 2), int(step_size / 2 + np.ceil(n / step_size) * step_size - n)

    if padding != 0:
        pad_left, pad_right = calculate_pad(len(inputs))
        inputs = np.concatenate([inputs[:1]] * pad_left + [inputs] + [inputs[-1:]] * pad_right, axis=0)
    progress = tqdm if verbose else lambda x: x
    for i in progress(range(0, len(inputs) - stride * (batch_size + 1), stride * batch_size)):
        index = (np.arange(batch_size) * stride + i)[:, None] + np.arange(kernel_size)[None]
        yield inputs[index].transpose(0, 4, 1, 2, 3)
