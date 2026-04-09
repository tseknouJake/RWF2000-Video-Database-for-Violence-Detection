from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from rwf2000_repro.common import color_jitter, normalize_clip, random_flip, uniform_sampling


ARCHITECTURE_CHANNELS = {
    "rgb": slice(0, 3),
    "opt": slice(3, 5),
    "flow-gated": slice(0, 5),
}


@dataclass(frozen=True)
class DatasetSummary:
    num_files: int
    class_names: tuple[str, ...]


class VideoSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        directory: str | Path,
        architecture: str,
        batch_size: int = 1,
        shuffle: bool = True,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.architecture = architecture
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.file_paths, self.label_map, self.class_names = self._scan_dataset()
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _scan_dataset(self) -> tuple[list[Path], dict[Path, np.ndarray], tuple[str, ...]]:
        if not self.directory.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.directory}")

        class_dirs = sorted(path for path in self.directory.iterdir() if path.is_dir())
        if not class_dirs:
            raise FileNotFoundError(f"No class subdirectories found in: {self.directory}")

        class_names = tuple(path.name for path in class_dirs)
        one_hots = tf.keras.utils.to_categorical(range(len(class_dirs)))
        file_paths: list[Path] = []
        label_map: dict[Path, np.ndarray] = {}

        for class_index, class_dir in enumerate(class_dirs):
            for file_path in sorted(class_dir.glob("*.npy")):
                file_paths.append(file_path)
                label_map[file_path] = one_hots[class_index]

        if not file_paths:
            raise FileNotFoundError(f"No .npy files found under: {self.directory}")

        return file_paths, label_map, class_names

    @property
    def summary(self) -> DatasetSummary:
        return DatasetSummary(num_files=len(self.file_paths), class_names=self.class_names)

    def __len__(self) -> int:
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_paths = [self.file_paths[item] for item in batch_indexes]
        batch_x = [self._load_clip(path) for path in batch_paths]
        batch_y = [self.label_map[path] for path in batch_paths]
        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _load_clip(self, path: Path) -> np.ndarray:
        data = np.load(path, mmap_mode="r")
        data = np.asarray(data, dtype=np.float32)
        data = uniform_sampling(data, target_frames=64)

        if self.architecture == "rgb":
            data = data[..., ARCHITECTURE_CHANNELS["rgb"]]
            if self.augment:
                data = color_jitter(data, self.rng)
                data = random_flip(data, probability=0.5, rng=self.rng)
            data = normalize_clip(data)
            return data

        if self.architecture == "opt":
            data = data[..., ARCHITECTURE_CHANNELS["opt"]]
            data = normalize_clip(data)
            return data

        if self.augment:
            data[..., :3] = color_jitter(data[..., :3], self.rng)
            data = random_flip(data, probability=0.5, rng=self.rng)

        data[..., :3] = normalize_clip(data[..., :3])
        data[..., 3:] = normalize_clip(data[..., 3:])
        return data


def make_learning_rate_scheduler(initial_learning_rate: float, decay_factor: float):
    def scheduler(epoch: int, current_lr: float) -> float:
        del current_lr
        if epoch != 0 and epoch % 10 == 0:
            return initial_learning_rate * (decay_factor ** (epoch // 10))
        return initial_learning_rate * (decay_factor ** (epoch // 10))

    return tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
