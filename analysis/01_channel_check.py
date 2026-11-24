"""Check channel identities (difference channels) and dropout patterns."""
import sys
from pathlib import Path
import numpy as np

A = Path(sys.argv[1]) / "Rehab_assessment"
P = A / "d02_processed_data"
R = A / "d01_raw_data"

# 1) which glove column = pitch3 - pitch1 ? which S2 column = pitch2 - pitch1 ?
err2, err3a, err3b = [], [], []
for k in range(1, 121, 7):
    for m in range(27):
        f = P / f"{k:03d}"
        s1 = np.load(f / f"{m:03d}_1.npy"); s2 = np.load(f / f"{m:03d}_2.npy"); s3 = np.load(f / f"{m:03d}_3.npy")
        sc = np.std(s2[:, 0]) + 1e-9
        err2.append(np.std(s2[:, 0] - (s2[:, 1] - s1[:, 0])) / sc)
        if np.std(s3[:, 0]) > 1 and np.std(s3[:, 1]) > 1:
            err3a.append(np.std(s3[:, 1] - (s3[:, 0] - s1[:, 0])) / np.std(s3[:, 1]))  # col1 = col0 - p1
            err3b.append(np.std(s3[:, 0] - (s3[:, 1] - s1[:, 0])) / np.std(s3[:, 0]))  # col0 = col1 - p1
print("S2 col0 == col1 - pitch1 : median rel err", np.median(err2).round(3))
print("glove col1 == col0 - pitch1:", np.median(err3a).round(3) if err3a else None,
      "| glove col0 == col1 - pitch1:", np.median(err3b).round(3) if err3b else None, f"(n={len(err3a)})")

# 2) dropout: S2 null rows (mag == -1 on all axes), glove all-zero files
null_s2, zero_glove = np.zeros(27), np.zeros(27)
n = 0
for k in range(1, 223):
    f = R / f"{k:03d}"
    n += 1
    for m in range(27):
        s2 = np.load(f / f"{m:03d}_2.npy"); s3 = np.load(f / f"{m:03d}_3.npy")
        null_s2[m] += np.mean(np.all(s2[:, 7:10] == -1, axis=1))
        zero_glove[m] += float(np.all(s3 == 0))
print("\nmovement  meanFrac_S2_null  frac_glove_allzero")
for m in range(27):
    print(f"{m:03d}      {null_s2[m]/n:.2f}             {zero_glove[m]/n:.2f}")

# 3) gyro magnitude scale (rad/s vs deg/s?)
g = np.concatenate([np.load(R / f"{k:03d}" / "026_1.npy")[:, 9:12] for k in range(1, 223, 5)])
print("\nS1 gyro |w| percentiles 50/95/99:", np.percentile(np.linalg.norm(g, axis=1), [50, 95, 99]).round(2))
a = np.concatenate([np.load(R / f"{k:03d}" / "026_1.npy")[:, 3:6] for k in range(1, 223, 5)])
print("S1 acc |a| median:", np.median(np.linalg.norm(a, axis=1)).round(3))
