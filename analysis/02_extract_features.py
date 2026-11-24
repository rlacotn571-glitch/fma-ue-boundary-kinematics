"""Kinematic feature extraction for REHAB assessment movements.

Usage: python -I 02_extract_features.py <extracted/REHAB dir> <out csv>

Uses raw data (d01, ~50 Hz) so that durations and velocities keep physical meaning.
Facts checked in the data (2026-11-23):
  * movements 000-014, 026: forearm (S1, file _1) + upper arm (S2, file _2); glove off
  * movements 015-025: glove only (file _3); S1 and S2 are zero
  * gyro values are deg/s (paper table says rad/s; ±250 deg/s range saturates near 250)
  * yaw wraps at ±180 deg -> unwrap before computing ranges
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

FS = 50.0
ARM = list(range(0, 15)) + [26]
GLOVE = list(range(15, 26))


def unwrap_deg(x):
    return np.degrees(np.unwrap(np.radians(x)))


def rng(x):
    """Robust range: 95th - 5th percentile."""
    if len(x) < 2:
        return np.nan
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def sparc(speed, fs=FS, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (Balasubramanian 2015). More negative = less smooth."""
    speed = np.asarray(speed, float)
    if len(speed) < 8 or np.max(speed) <= 0:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / Mf.max()
    sel = f <= fc
    f, Mf = f[sel], Mf[sel]
    idx = np.nonzero(Mf >= amp_th)[0]
    if len(idx) < 2:
        return np.nan
    f, Mf = f[idx[0]: idx[-1] + 1], Mf[idx[0]: idx[-1] + 1]
    df = np.diff(f) / (f[-1] - f[0])
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(Mf) ** 2)))


def n_peaks(speed):
    s = np.convolve(speed, np.ones(5) / 5, mode="same")
    if len(s) < 3 or s.max() <= 0:
        return np.nan
    th = 0.1 * s.max()
    pk = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] > th)
    return int(pk.sum())


def imu_feats(x, tag, angle_cols, gyro_cols):
    out = {}
    ang = np.column_stack([unwrap_deg(x[:, c]) for c in angle_cols])
    for name, a in zip(["pitch", "yaw", "roll"], ang.T):
        out[f"{tag}_{name}_rom"] = rng(a)
    out[f"{tag}_rom_max"] = np.nanmax([out[f"{tag}_{n}_rom"] for n in ["pitch", "yaw", "roll"]])
    w = np.linalg.norm(x[:, gyro_cols], axis=1)
    out[f"{tag}_w_peak"] = float(np.percentile(w, 95)) if len(w) else np.nan
    out[f"{tag}_w_mean"] = float(np.mean(w)) if len(w) else np.nan
    out[f"{tag}_sparc"] = sparc(w)
    out[f"{tag}_npeaks"] = n_peaks(w)
    out[f"{tag}_flat"] = bool(np.all(np.std(x, axis=0) < 1e-3))
    return out


def glove_feats(g):
    out = {}
    pa, pb = unwrap_deg(g[:, 0]), unwrap_deg(g[:, 1])
    out["g_col0_rom"], out["g_col1_rom"] = rng(pa), rng(pb)
    # wrist pitch: use the column with fewer large sample-to-sample jumps
    ja, jb = np.sum(np.abs(np.diff(pa)) > 30), np.sum(np.abs(np.diff(pb)) > 30)
    p = pb if jb <= ja else pa
    out["g_pitch_rom"] = rng(p)
    out["g_pitch_jumps"] = int(min(ja, jb))
    v = np.abs(np.diff(p)) * FS
    out["g_w_peak"] = float(np.percentile(v, 95)) if len(v) else np.nan
    out["g_sparc"] = sparc(v)
    out["g_npeaks"] = n_peaks(v) if len(v) else np.nan
    for c, n in zip([2, 3, 4], ["f1", "f12", "fall"]):
        out[f"{n}_rom"] = rng(g[:, c])
        out[f"{n}_max"] = float(np.percentile(g[:, c], 95))
        out[f"{n}_min"] = float(np.percentile(g[:, c], 5))
    out["g_flat"] = bool(np.all(np.std(g, axis=0) < 1e-3))
    return out


# a priori primary feature per item (target joint of each FMA item)
PRIMARY = {
    0: "S2_rom_max", 1: "S2_rom_max", 2: "S2_rom_max", 3: "S1_rom_max",
    4: "elbow_rom", 5: "S1_roll_rom", 6: "S1_rom_max", 7: "elbow_rom", 8: "S1_roll_rom",
    9: "S1_rom_max", 10: "S2_pitch_rom", 11: "S1_roll_rom", 12: "S1_roll_rom",
    13: "S1_roll_rom", 14: "S1_roll_rom",
    15: "g_pitch_rom", 16: "g_pitch_rom", 17: "g_pitch_rom", 18: "g_pitch_rom", 19: "g_pitch_rom",
    20: "fall_rom", 21: "fall_rom", 22: "f1_rom", 23: "f12_rom", 24: "fall_rom", 25: "fall_rom",
    26: "S1_sparc",
}


def main():
    root = Path(sys.argv[1]) / "Rehab_assessment"
    out_csv = Path(sys.argv[2])
    lab = np.load(root / "Label.npy", allow_pickle=False)
    R = root / "d01_raw_data"
    rows = []
    for k in range(1, 223):
        for m in range(27):
            f = R / f"{k:03d}"
            row = {"assess": k, "move": m, "score": int(lab[k - 1, m])}
            if m in ARM:
                s1 = np.load(f / f"{m:03d}_1.npy", allow_pickle=False)
                s2 = np.load(f / f"{m:03d}_2.npy", allow_pickle=False)
                row["n1"], row["n2"] = len(s1), len(s2)
                row["dur_s"] = max(len(s1), len(s2)) / FS
                row.update(imu_feats(s1, "S1", [0, 1, 2], [9, 10, 11]))
                row.update(imu_feats(s2, "S2", [1, 2, 3], [10, 11, 12]))
                row["elbow_rom"] = rng(unwrap_deg(s2[:, 0]))
                row["comp_ratio"] = row["S2_rom_max"] / (row["S1_rom_max"] + 1.0)
            else:
                g = np.load(f / f"{m:03d}_3.npy", allow_pickle=False)
                row["n3"] = len(g)
                row["dur_s"] = len(g) / FS
                row.update(glove_feats(g))
            row["primary"] = row.get(PRIMARY[m], np.nan)
            row["primary_name"] = PRIMARY[m]
            row["short"] = row["dur_s"] < 0.5
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved", out_csv, df.shape)


if __name__ == "__main__":
    main()
