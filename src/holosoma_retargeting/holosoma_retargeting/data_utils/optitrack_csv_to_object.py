"""Convert an OptiTrack (Motive) CSV export into an object-pose .npy for object_interaction.

The dice in the capture is tracked as several rigid bodies (e.g. ``Dice corner 1/2/3``),
each rigidly attached to the physical dice. This script:

  1. Reads the position of every matching rigid body each frame.
  2. Uses Kabsch alignment of those points (reference = first frame) to recover a clean,
     gap-free 6-DoF pose of the dice (centroid + rotation). A cube is rotationally
     symmetric, so the orientation only needs to be consistent, not face-exact.
  3. Optionally converts a Y-up export to Z-up and applies a constant body-frame offset
     to move the tracked centroid onto the true geometric center of the dice.
  4. Downsamples to match the human SMPL-X sequence and writes ``(T, 7)`` =
     ``[qw, qx, qy, qz, x, y, z]`` in the SAME world frame/units as the human.

Output is consumed by ``examples/robot_retarget.py`` (object_interaction, ``--data_format smplx``)
as ``<data_path>/<task_name>_object.npy``.

Example:
    python data_utils/optitrack_csv_to_object.py \
        --csv /path/to/Traj1.csv \
        --output demo_data/dice/my_dice_seq_object.npy \
        --rigid-body-prefix "Dice corner" --downsample 2
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R


@dataclass
class Config:
    csv: str
    """Path to the OptiTrack/Motive CSV export."""

    output: str
    """Output .npy path, shape (T, 7) = [qw, qx, qy, qz, x, y, z]."""

    rigid_body_prefix: str = "Dice corner"
    """Name prefix selecting the rigid bodies rigidly attached to the dice."""

    downsample: int = 2
    """Keep every Nth frame. MUST match the --downsample used for the human SMPL-X prep."""

    yup_to_zup: bool = False
    """Set True only if the CSV is Y-up. This Motive export is already Z-up (Meters)."""

    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Constant offset (meters, dice body frame) added to the tracked centroid to reach the
    true dice center. Leave at 0 and refine by eye in the viser viewer if the cube looks
    shifted relative to the robot hands."""

    header_rows: int = 7
    """Number of header lines before the first data row in the Motive export."""


def _find_rigid_body_position_columns(csv_path: str, prefix: str) -> dict[str, tuple[int, int, int]]:
    """Return {rigid_body_name: (col_x, col_y, col_z)} for every matching rigid body."""
    rows = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            rows.append(row)
            if i >= 6:
                break
    type_row, name_row, chan_row, axis_row = rows[2], rows[3], rows[5], rows[6]

    out: dict[str, tuple[int, int, int]] = {}
    n = len(name_row)
    j = 0
    while j < n:
        name = name_row[j].strip()
        if not name:
            j += 1
            continue
        k = j
        while k < n and name_row[k] == name_row[j]:
            k += 1
        if type_row[j].strip() == "Rigid Body" and name.startswith(prefix):
            # Within this block, find the three Position channels and their X/Y/Z axes.
            pos = {}
            for c in range(j, k):
                if chan_row[c].strip() == "Position":
                    pos[axis_row[c].strip()] = c
            out[name] = (pos["X"], pos["Y"], pos["Z"])
        j = k
    if not out:
        raise ValueError(f"No rigid bodies with prefix '{prefix}' found in {csv_path}")
    return out


def _load_data(csv_path: str, header_rows: int) -> np.ndarray:
    data = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i < header_rows:
                continue
            data.append([(float(x) if x not in ("", "nan") else np.nan) for x in row])
    return np.asarray(data, dtype=float)


def _forward_fill(a: np.ndarray) -> np.ndarray:
    """Fill NaNs forward (then backward) along time for a (T, ...) array."""
    out = a.copy()
    for t in range(1, out.shape[0]):
        m = np.isnan(out[t])
        out[t][m] = out[t - 1][m]
    for t in range(out.shape[0] - 2, -1, -1):
        m = np.isnan(out[t])
        out[t][m] = out[t + 1][m]
    return out


def _kabsch(ref_c: np.ndarray, cur_c: np.ndarray) -> np.ndarray:
    """Rotation R such that cur_c ~= (R @ ref_c.T).T, for centered point sets (N,3)."""
    h = ref_c.T @ cur_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def main(cfg: Config) -> None:
    cols = _find_rigid_body_position_columns(cfg.csv, cfg.rigid_body_prefix)
    names = sorted(cols)
    print(f"Found {len(names)} rigid bodies: {names}")

    arr = _load_data(cfg.csv, cfg.header_rows)
    # (T, N, 3) positions for each tracked corner.
    pts = np.stack([arr[:, list(cols[nm])] for nm in names], axis=1)
    pts = _forward_fill(pts)
    n_frames = pts.shape[0]

    if cfg.yup_to_zup:
        # (x, y, z)_zup = (x, z, -y) for a Y-up -> Z-up change of basis.
        pts = np.stack([pts[..., 0], pts[..., 2], -pts[..., 1]], axis=-1)

    ref = pts[0]
    ref_c = ref - ref.mean(axis=0)
    offset = np.asarray(cfg.center_offset, dtype=float)

    poses = np.zeros((n_frames, 7), dtype=float)
    for t in range(n_frames):
        cur = pts[t]
        centroid = cur.mean(axis=0)
        rot = _kabsch(ref_c, cur - centroid)
        quat_xyzw = R.from_matrix(rot).as_quat()  # [qx, qy, qz, qw]
        poses[t, :4] = quat_xyzw[[3, 0, 1, 2]]  # -> [qw, qx, qy, qz]
        poses[t, 4:7] = centroid + rot @ offset

    poses = poses[:: cfg.downsample]
    np.save(cfg.output, poses)
    print(
        f"Saved {poses.shape[0]} frames -> {cfg.output}\n"
        f"  pos[0]={np.round(poses[0, 4:7], 3)}  quat[0]={np.round(poses[0, :4], 3)}\n"
        f"  pos range min={np.round(poses[:, 4:7].min(0), 3)} max={np.round(poses[:, 4:7].max(0), 3)}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
