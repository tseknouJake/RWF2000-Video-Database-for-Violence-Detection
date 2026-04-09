from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if is_video_file(path):
            yield path


def read_video_frames(file_path: Path, resize: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(file_path))
    frames: list[np.ndarray] = []

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.reshape(resize[1], resize[0], 3))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Video contains no readable frames: {file_path}")

    return np.asarray(frames, dtype=np.uint8)


def get_optical_flow(video: np.ndarray) -> np.ndarray:
    """Mirror the notebook preprocessing: dense Farneback optical flow per frame."""
    gray_video = []
    height = video.shape[1]
    width = video.shape[2]

    for frame in video:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_video.append(gray.reshape(height, width))

    flows = []
    for index in range(0, len(video) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_video[index],
            gray_video[index + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)

    flows.append(np.zeros((height, width, 2), dtype=np.float32))
    return np.asarray(flows, dtype=np.float32)


def video_to_tensor(file_path: Path, resize: tuple[int, int] = (224, 224)) -> np.ndarray:
    frames = read_video_frames(file_path=file_path, resize=resize)
    flows = get_optical_flow(frames)

    result = np.zeros((len(flows), resize[1], resize[0], 5), dtype=np.float32)
    result[..., :3] = frames
    result[..., 3:] = flows
    return np.asarray(result, dtype=np.uint8)


def uniform_sampling(video: np.ndarray, target_frames: int = 64) -> np.ndarray:
    len_frames = int(len(video))
    if len_frames == 0:
        raise ValueError("Cannot sample from an empty video.")

    interval = max(1, int(math.ceil(len_frames / target_frames)))
    sampled_video = [video[index] for index in range(0, len_frames, interval)]

    if len(sampled_video) > target_frames:
        sampled_video = sampled_video[:target_frames]

    num_pad = target_frames - len(sampled_video)
    if num_pad > 0:
        padding = [sampled_video[-1] for _ in range(num_pad)]
        sampled_video.extend(padding)

    return np.asarray(sampled_video, dtype=np.float32)


def normalize_clip(data: np.ndarray) -> np.ndarray:
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std < 1e-6:
        std = 1.0
    return (data - mean) / std


def random_flip(video: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < probability:
        return np.flip(video, axis=2).copy()
    return video


def color_jitter(video: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    video = video.copy()
    s_jitter = rng.uniform(-0.2, 0.2)
    v_jitter = rng.uniform(-30.0, 30.0)

    for index in range(len(video)):
        hsv = cv2.cvtColor(video[index], cv2.COLOR_RGB2HSV)
        saturation = hsv[..., 1] + s_jitter
        value = hsv[..., 2] + v_jitter
        saturation[saturation < 0] = 0
        saturation[saturation > 1] = 1
        value[value < 0] = 0
        value[value > 255] = 255
        hsv[..., 1] = saturation
        hsv[..., 2] = value
        video[index] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return video
