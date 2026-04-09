from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained or pretrained RWF-2000 model.")
    parser.add_argument("--model-path", required=True, help="Path to a .keras or .h5 checkpoint.")
    parser.add_argument("--data-dir", required=True, help="Processed validation or test directory.")
    parser.add_argument("--architecture", choices=("rgb", "opt", "flow-gated"), default="flow-gated")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", default="", help="Optional path for a JSON metrics file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        import tensorflow as tf

        from rwf2000_repro.models import load_model
        from rwf2000_repro.training import VideoSequence
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is not installed. Create the Python 3.11 virtualenv from REPRODUCE.md and run `pip install -r requirements.txt` first."
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)

    data_sequence = VideoSequence(
        directory=args.data_dir,
        architecture=args.architecture,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        seed=args.seed,
    )
    model = load_model(args.model_path, architecture=args.architecture)
    predictions = model.predict(data_sequence, verbose=1)

    y_true = []
    for index in range(len(data_sequence)):
        _, labels = data_sequence[index]
        y_true.extend(np.argmax(labels, axis=1))
    y_true = np.asarray(y_true[: len(predictions)])

    y_pred = np.argmax(predictions, axis=1)
    accuracy = float(np.mean(y_true == y_pred))
    confusion = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    metrics = {
        "architecture": args.architecture,
        "model_path": str(Path(args.model_path).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "num_samples": int(len(y_true)),
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(data_sequence.class_names),
    }

    print(json.dumps(metrics, indent=2))

    if args.metrics_out:
        metrics_out = Path(args.metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
