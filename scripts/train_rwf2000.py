from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one of the RWF-2000 notebook baselines.")
    parser.add_argument("--architecture", choices=("rgb", "opt", "flow-gated"), default="flow-gated")
    parser.add_argument("--train-dir", required=True, help="Processed training split directory.")
    parser.add_argument("--val-dir", required=True, help="Processed validation split directory.")
    parser.add_argument("--output-dir", default="outputs/flow-gated", help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def decay_factor_for_architecture(architecture: str) -> float:
    if architecture == "flow-gated":
        return 0.7
    return 0.5


def build_compiled_model(tf, build_model, architecture: str, learning_rate: float):
    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = build_model(architecture)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        )
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    return strategy, model


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf

        from rwf2000_repro.models import build_model
        from rwf2000_repro.training import VideoSequence, make_learning_rate_scheduler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is not installed. Create the Python 3.11 virtualenv from REPRODUCE.md and run `pip install -r requirements.txt` first."
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)

    train_sequence = VideoSequence(
        directory=args.train_dir,
        architecture=args.architecture,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        seed=args.seed,
    )
    val_sequence = VideoSequence(
        directory=args.val_dir,
        architecture=args.architecture,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        seed=args.seed,
    )

    _, model = build_compiled_model(tf, build_model, args.architecture, args.learning_rate)
    scheduler = make_learning_rate_scheduler(
        initial_learning_rate=args.learning_rate,
        decay_factor=decay_factor_for_architecture(args.architecture),
    )
    csv_logger = tf.keras.callbacks.CSVLogger(output_dir / "train_log.csv")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output_dir / "best_model.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=[scheduler, csv_logger, checkpoint],
        verbose=1,
    )

    model.save(output_dir / "final_model.keras")
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)

    print(f"Saved checkpoints and logs to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
