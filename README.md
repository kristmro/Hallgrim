# Hallgrim

Simple utility for inspecting `.npz` files and plotting every contained array against time.

## Usage

```bash
python inspect_npz.py data.npz --output-dir plots
```

### Options

- `--time-key TIME_ARRAY_NAME`: choose which array to use as the x-axis.
  - If omitted, the script uses `time` when available.
  - Otherwise it uses `0..N-1` indices.
- `--skip NAME [NAME ...]`: skip plotting specific arrays.
- `--output-dir PATH`: save each plot as PNG instead of opening interactive windows.

The script always prints array names, shapes, and dtypes before plotting.
