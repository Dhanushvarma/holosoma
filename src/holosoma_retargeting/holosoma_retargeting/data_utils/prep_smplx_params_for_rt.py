"""FK an SMPL-X parameter sequence into the retargeting input format.

Input: an .npz with SMPL-X body parameters (the AMASS-style / GMR layout)::

    pose_body         (T, 63)   body-joint axis-angle (21 joints)
    root_orient       (T, 3)    global orientation axis-angle (pelvis)
    trans             (T, 3)    root translation (meters, world frame)
    betas             (16,)     shape parameters
    gender            scalar    'male' | 'female' | 'neutral'
    mocap_frame_rate  scalar    capture fps

Output: an .npz with ``global_joint_positions`` (T, 22, 3) and scalar ``height``, which is
exactly what ``examples/robot_retarget.py`` consumes for ``--data_format smplx``. The first
22 SMPL-X joints already match ``SMPLX_DEMO_JOINTS`` in config_types/data_type.py.

Requires the SMPL-X model file, e.g. ``<model_path>/smplx/SMPLX_NEUTRAL.npz`` (license-gated;
download from https://smpl-x.is.tue.mpg.de). The ``smplx`` package is already a dependency.

Example:
    python data_utils/prep_smplx_params_for_rt.py \
        --input /path/to/Traj1_smpl_gmr_latest_gmr.npz \
        --output demo_data/dice/my_dice_seq.npz \
        --model-path /path/to/smpl_models --downsample 2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import smplx
import torch
import tyro

NUM_BODY_JOINTS = 22  # matches SMPLX_DEMO_JOINTS


@dataclass
class Config:
    input: str
    """Path to the SMPL-X parameter .npz."""

    output: str
    """Output .npz path (global_joint_positions + height)."""

    model_path: str
    """Folder containing the SMPL-X model, i.e. <model_path>/smplx/SMPLX_<GENDER>.<ext>."""

    gender: str | None = None
    """Override gender. If None, use the 'gender' field stored in the input npz."""

    ext: str = "npz"
    """SMPL-X model file extension ('npz' or 'pkl')."""

    downsample: int = 2
    """Keep every Nth frame. MUST match --downsample used for the OptiTrack object prep."""

    num_betas: int = 16
    """Number of shape coefficients (this data uses 16)."""

    chunk: int = 256
    """Frames per forward pass (limits memory)."""


def _zero_pose_kwargs(model: smplx.SMPLX, batch: int) -> dict[str, torch.Tensor]:
    """Explicit zero hand/jaw/eye/expression poses sized to `batch`.

    Passing these makes the forward pass independent of the model's configured batch_size,
    so partial chunks don't clash with the model's default pose buffers. None of them affect
    the first 22 (body) joints we keep.
    """
    z = lambda d: torch.zeros(batch, d, dtype=torch.float32)  # noqa: E731
    return {
        "left_hand_pose": z(model.num_pca_comps if model.use_pca else 45),
        "right_hand_pose": z(model.num_pca_comps if model.use_pca else 45),
        "jaw_pose": z(3),
        "leye_pose": z(3),
        "reye_pose": z(3),
        "expression": z(model.num_expression_coeffs),
    }


def _pad(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Pad along dim 0 up to `batch` by repeating the last row (no-op if already full)."""
    if x.shape[0] == batch:
        return x
    return torch.cat([x, x[-1:].expand(batch - x.shape[0], *x.shape[1:])], dim=0)


def main(cfg: Config) -> None:
    data = np.load(cfg.input, allow_pickle=True)
    gender = cfg.gender or str(data["gender"])
    print(f"gender={gender}  frames={data['trans'].shape[0]}  fps={float(data['mocap_frame_rate'])}")

    s = slice(None, None, cfg.downsample)
    root_orient = torch.as_tensor(np.asarray(data["root_orient"])[s], dtype=torch.float32)
    pose_body = torch.as_tensor(np.asarray(data["pose_body"])[s], dtype=torch.float32)
    trans = torch.as_tensor(np.asarray(data["trans"])[s], dtype=torch.float32)
    betas = torch.as_tensor(np.asarray(data["betas"])[: cfg.num_betas], dtype=torch.float32)[None]
    T = trans.shape[0]

    # The smplx model's internal LBS buffers are pinned to batch_size, so we keep a fixed
    # batch and pad the final partial chunk (then trim the output).
    batch = min(cfg.chunk, T)
    model = smplx.create(
        model_path=cfg.model_path,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=cfg.num_betas,
        ext=cfg.ext,
        batch_size=batch,
    )
    zero_poses = _zero_pose_kwargs(model, batch)

    joints = np.empty((T, NUM_BODY_JOINTS, 3), dtype=np.float32)
    for i in range(0, T, batch):
        j = min(i + batch, T)
        out = model(
            global_orient=_pad(root_orient[i:j], batch),
            body_pose=_pad(pose_body[i:j], batch),
            betas=betas.expand(batch, -1),
            transl=_pad(trans[i:j], batch),
            return_verts=False,
            **zero_poses,
        )
        joints[i:j] = out.joints[: j - i, :NUM_BODY_JOINTS, :].detach().cpu().numpy()

    # Height from the shaped rest-pose mesh (canonical up-axis = Y), reusing the same model.
    rest = model(betas=betas.expand(batch, -1), return_verts=True, **zero_poses)
    v = rest.vertices[0].detach().cpu().numpy()
    height = float(v[:, 1].max() - v[:, 1].min())
    print(f"height={height:.3f} m   joints shape={joints.shape}")
    print(f"feet z-min={joints[:, [10, 11], 2].min():.3f}  pelvis z[0]={joints[0, 0, 2]:.3f}")

    np.savez(cfg.output, global_joint_positions=joints, height=height)
    print(f"Saved -> {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
