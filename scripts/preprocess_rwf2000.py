from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwf2000_repro.preprocessing import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RWF-2000 videos into 5-channel .npy tensors.")
    parser.add_argument("--source-dir", required=True, help="Root directory containing the raw video dataset.")
    parser.add_argument("--target-dir", required=True, help="Output directory for processed .npy files.")
    parser.add_argument("--resize", type=int, nargs=2, default=(224, 224), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    converted, skipped = preprocess_dataset(
        source_root=Path(args.source_dir),
        target_root=Path(args.target_dir),
        resize=tuple(args.resize),
        overwrite=args.overwrite,
    )
    print(f"Converted {converted} videos. Skipped {skipped} existing files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
