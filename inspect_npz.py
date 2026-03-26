#!/usr/bin/env python3
"""Start-simple NPZ inspector: set path, run script, list objects."""

from pathlib import Path

import numpy as np

# 1) Put your .npz path here.
NPZ_PATH = Path("/absolute/path/to/your_file.npz")


def main() -> None:
    npz_file = NPZ_PATH
    if not npz_file.exists():
        print(f"NPZ file not found: {npz_file}")
        print("Edit NPZ_PATH in inspect_npz.py and run again.")
        return

    with np.load(npz_file) as archive:
        print(f"Loaded: {npz_file}")
        print("Data objects found:")
        for key in archive.files:
            arr = archive[key]
            print(f"- {key} (shape={arr.shape}, dtype={arr.dtype})")


if __name__ == "__main__":
    main()
