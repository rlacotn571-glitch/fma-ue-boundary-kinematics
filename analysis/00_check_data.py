"""Sanity check of REHAB assessment data (paths passed explicitly, data dir is untrusted)."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np

root = Path(sys.argv[1])  # .../extracted/REHAB
A = root / "Rehab_assessment"

lab = np.load(A / "Label.npy", allow_pickle=False)
print("Label", lab.shape, lab.dtype, "values", sorted(np.unique(lab).tolist()))

for sub in ["d01_raw_data", "d02_processed_data"]:
    shapes = {1: Counter(), 2: Counter(), 3: Counter()}
    lengths, missing, bad = [], [], []
    for k in range(1, 223):
        d = A / sub / f"{k:03d}"
        files = list(d.rglob("*.npy"))
        if len(files) != 81:
            missing.append((k, len(files)))
        for f in files:
            try:
                x = np.load(f, allow_pickle=False)
            except Exception as e:
                bad.append((f.relative_to(A).as_posix(), str(e)[:60]))
                continue
            part = int(f.stem.split("_")[1])
            shapes[part][x.shape[1] if x.ndim == 2 else x.shape] += 1
            if part == 1:
                lengths.append(x.shape[0])
    print(f"\n[{sub}] folders with !=81 files: {missing[:10]} ({len(missing)})")
    print("  channel counts:", {p: dict(c) for p, c in shapes.items()})
    if lengths:
        L = np.array(lengths)
        print(f"  _1 length: min {L.min()} median {int(np.median(L))} max {L.max()}")
    print("  unreadable:", len(bad), bad[:5])

# exercise file flagged earlier
ex = root / "Rehab_exercise" / "d02_processed_data" / "014_1.npy"
if ex.exists():
    try:
        x = np.load(ex, allow_pickle=False)
        print("\n014_1", x.shape, x.dtype)
    except Exception as e:
        print("\n014_1 unreadable:", str(e)[:120])

# quick partial-credit Rasch on folders 001-120
from girth import pcm_mml
res = pcm_mml(lab[:120].T.astype(int))
thr = np.asarray(res["Difficulty"])  # items x thresholds
print("\nPCM (n=120) mean threshold per item (higher = harder):")
order = np.argsort(np.nanmean(thr, axis=1))
print(" ".join(f"{i:03d}:{np.nanmean(thr[i]):+.2f}" for i in order))
