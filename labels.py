from pathlib import Path
import numpy as np

NPZ_PATH = Path(r"C:\Users\kmroe\Desktop\5 klasse\hallgrim\S1_30T06_16_R1_LANG_0102.npz")

with np.load(NPZ_PATH, allow_pickle=True) as archive:
    print("Top-level entries:", list(archive.files))
    if "keys" in archive:
        keys = archive["keys"]
        # decode bytes if necessary
        if keys.dtype.kind in {"S", "O"}:
            keys = [k.decode() if isinstance(k, bytes) else str(k) for k in keys]
        print("\nLabels inside 'keys':")
        for i, label in enumerate(keys, start=0):
            print(f"{i}: {label}")
    else:
        print("No 'keys' array found. Inspecting 'data' object:")
        data_obj = archive["data"].item() if archive["data"].shape == () else archive["data"]
        if isinstance(data_obj, dict):
            for i, k in enumerate(data_obj.keys()):
                print(f"{i}: {k}")
        else:
            print("Unexpected 'data' format:", type(data_obj))