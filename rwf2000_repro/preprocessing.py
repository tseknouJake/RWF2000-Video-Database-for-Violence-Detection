from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from rwf2000_repro.common import iter_video_files, video_to_tensor


def preprocess_dataset(
    source_root: Path,
    target_root: Path,
    resize: tuple[int, int] = (224, 224),
    overwrite: bool = False,
) -> tuple[int, int]:
    source_root = source_root.resolve()
    target_root = target_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_root}")

    videos = list(iter_video_files(source_root))
    if not videos:
        raise FileNotFoundError(f"No video files found under: {source_root}")

    converted = 0
    skipped = 0

    for video_path in tqdm(videos, desc="Preprocessing videos"):
        relative_path = video_path.relative_to(source_root).with_suffix(".npy")
        output_path = target_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        tensor = video_to_tensor(video_path, resize=resize)
        np.save(output_path, tensor)
        converted += 1

    return converted, skipped
