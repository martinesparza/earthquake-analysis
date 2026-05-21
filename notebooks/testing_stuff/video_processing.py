"""
Test script: extract first 3 perturbation trials from Camera_2,
concatenate them, and overlay a red dot (turns green at perturbation onset).
"""

import sys
import os

sys.path.insert(0, "/home/me24/repos/earthquake-analysis")

import numpy as np
import pyaldata as pyal
import cv2
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "/data/bnd-data/raw/"
SESSION = "M106_2026_02_25_15_00"
VIDEO_PATH = f"{DATA_DIR}M106/{SESSION}/{SESSION}_cameras/Camera_1.avi"
RESULTS_DIR = f"{DATA_DIR}M106/{SESSION}/{SESSION}_cameras/"

# Video/data are both at 100 fps/Hz — 1:1 frame ↔ bin mapping
FPS = 100
DOT_DURATION = 20  # frames the dot stays green after perturbation onset
DOT_POS = None  # set after reading video dimensions
DOT_RADIUS = 40
N_TRIALS = None  # None = all trials

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading pyaldata...")
df = pyal.load_pyaldata(DATA_DIR + "M106/" + SESSION)

# ── Build absolute frame index per row ────────────────────────────────────
# Neural bin count = video frame count (both at 100 Hz / 100 fps)
area = next(c for c in df.columns if c.endswith("_spikes"))
row_lengths = [arr.shape[0] for arr in df[area].values]
row_frame_starts = np.concatenate([[0], np.cumsum(row_lengths[:-1])])

print(f"Total rows: {len(df)}, total frames: {sum(row_lengths)}")
print(f"At 100fps → {sum(row_lengths)/100:.1f}s")

# ── Collect first N_TRIALS perturbation-trial segments ────────────────────
segments = []  # (abs_start, abs_end, perturb_abs_frame)
for i, (_, row) in enumerate(df.iterrows()):
    if row.trial_name != "trial":
        continue
    sol_on = row.idx_sol_on
    if not np.isscalar(sol_on):  # intertrial rows have empty array
        continue
    abs_start = int(row_frame_starts[i])
    abs_end = abs_start + int(row_lengths[i])
    perturb = abs_start + int(sol_on)
    sol_dir = int(row.values_Sol_direction) if np.isscalar(row.values_Sol_direction) else -1
    segments.append((abs_start, abs_end, perturb, int(row.trial_id), sol_dir))
    print(
        f"  trial_id={row.trial_id}  sol_dir={sol_dir}  frames {abs_start}–{abs_end}  "
        f"({abs_end-abs_start} frames)  perturbation @ frame {perturb}"
    )
    if N_TRIALS is not None and len(segments) == N_TRIALS:
        break

# ── Open source video ──────────────────────────────────────────────────────
print(f"\nOpening {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}×{height}, {total_frames} total frames")
DOT_POS = (width - 60, 60)

# ── Write output ───────────────────────────────────────────────────────────
out_avi = RESULTS_DIR + "Camera_1_trials.avi"

out = cv2.VideoWriter(out_avi, cv2.VideoWriter_fourcc(*"XVID"), FPS, (width, height))

total_output_frames = sum(e - s for s, e, *_ in segments)
print(f"\nTotal output frames: {total_output_frames} ({total_output_frames/FPS:.1f}s)")

for seg_idx, (abs_start, abs_end, perturb, trial_id, sol_dir) in enumerate(segments):
    print(f"\nSegment {seg_idx+1}/{len(segments)}: seeking to frame {abs_start}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_start)

    label = f"Sol {sol_dir}  Trial {trial_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_pos = (DOT_POS[0] - DOT_RADIUS - text_w - 15, DOT_POS[1] + text_h // 2)

    for frame_idx in tqdm(range(abs_start, abs_end), desc=f"Trial {trial_id}"):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: ran out of frames at {frame_idx}")
            break
        at_perturb = perturb <= frame_idx < perturb + DOT_DURATION
        colour = (0, 255, 0) if at_perturb else (0, 0, 255)
        cv2.circle(frame, DOT_POS, radius=DOT_RADIUS, color=colour, thickness=-1)
        cv2.putText(
            frame, label, text_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
        out.write(frame)

cap.release()
out.release()

print(f"Done! Saved to {out_avi}")
