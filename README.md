# Hallgrim

Very small starter utility to inspect a `.npz` file.

## What it does now

- You set the file path directly in `inspect_npz.py` (`NPZ_PATH`).
- You run the script.
- It prints all data objects inside the `.npz` (name, shape, dtype).

## Usage

1. Open `inspect_npz.py` and set:
   - `NPZ_PATH = Path("/absolute/path/to/your_file.npz")`
2. Run:

```bash
python inspect_npz.py
```

This is intentionally minimal so we can add features step by step.
