# Retargeting Custom Mocap (SMPL-X human + OptiTrack object) → G1

Object-interaction retargeting from **two split sources**: an SMPL-X parameter file for the
human, and an OptiTrack/Motive CSV for the manipulated object (a dice). Worked example:
`Traj1` (female subject, dice held in hands, 60 fps, 1501 frames).

## Data format

| Signal | Shape / format | Frame |
|---|---|---|
| Human joints | `global_joint_positions (T,22,3)` + scalar `height`, in a `.npz` | Z-up, meters |
| Object pose | `(T,7) = [qw,qx,qy,qz, x,y,z]`, in a `.npy` | Z-up, meters, **same world as human** |

Both must have **identical frame count** (the loader hard-checks this). The pipeline scales
human + object down to robot proportions internally — feed real human-scale meters.

## Pipeline

```
SMPL-X params ──FK──► Traj1.npz (human joints + height)
OptiTrack CSV ─Kabsch► Traj1_object.npy (dice pose)        ┐
                                                            ├─► robot_retarget.py ─► Traj1_original.npz
dice.stl ──► dice.obj/.urdf + g1_29dof_w_dice.xml          ┘     (object_interaction, --data_format smplx)
```

## One-time setup (code + assets)

These are already applied in this repo:

1. **`config_types/data_type.py`** — `smplx` registered in `DEMO_JOINTS_REGISTRY`; the
   `("smplx","g1")` mapping, `SMPLX_DEMO_JOINTS`, and `TOE_NAMES_BY_FORMAT["smplx"]` already existed.
2. **`examples/robot_retarget.py`** — `validate_config` accepts `smplx` for `object_interaction`;
   `load_motion_data` loads the human `.npz` + `<task>_object.npy`; `main` enables floor-relative
   object scaling for this path.
3. **`src/utils.py`** — `preprocess_motion_data` gained `object_scale_about_floor` (opt-in, default
   off so OMOMO is untouched): scales a **held/elevated** object with the same floor transform as
   the human, so it stays in the robot's hands after the ~0.8× shrink.
4. **Dice model** — `models/dice/dice.obj` (+ `.urdf`, `templates/dice.urdf.jinja`) and the combined
   scene `models/g1/g1_29dof_w_dice.xml` (copy of `g1_29dof_w_largebox.xml`, `largebox`→`dice`).

## Steps

### 1. Human: SMPL-X params → joints `.npz`
Needs the license-gated SMPL-X model at `<model_path>/smplx/SMPLX_<GENDER>.npz`.
```bash
python data_utils/prep_smplx_params_for_rt.py \
  --input  /path/to/Traj1_smpl_gmr_latest_gmr.npz \
  --output demo_data/dice/Traj1.npz \
  --model-path /path/to/smpl_models --gender female --downsample 2
```
FK's the first 22 SMPL-X joints (already in `SMPLX_DEMO_JOINTS` order) and computes height from the
shaped rest mesh. Sanity: feet `z-min ≈ 0` (Z-up), height plausible (~1.65 m here).

### 2. Object: OptiTrack CSV → dice pose `.npy`
```bash
python data_utils/optitrack_csv_to_object.py \
  --csv /path/to/Traj1.csv \
  --output demo_data/dice/Traj1_object.npy \
  --rigid-body-prefix "Dice corner" --downsample 2
```
The dice is tracked as 3 rigid bodies (`Dice corner 1/2/3`); Kabsch on their positions gives a clean,
gap-free 6-DoF pose (centroid + rotation). **`--downsample` must match step 1.** Motive quaternions
are `[X,Y,Z,W]` → reordered to `[qw,qx,qy,qz]`. This export is already Z-up/meters (`--yup-to-zup`
only if yours is Y-up).

### 3. Retarget to G1
```bash
python examples/robot_retarget.py \
  --task-type object_interaction --data_format smplx --robot g1 \
  --data_path demo_data/dice --task-name Traj1 \
  --task-config.object-name dice \
  --save_dir demo_results/g1/object_interaction/dice \
  --retargeter.debug --retargeter.visualize       # drop both flags to run headless
```
Per-frame interaction-mesh IK, ~1 s/frame. Output: `Traj1_original.npz` with
`qpos (T,43)` = 7 base + 29 G1 DoF + 7 dice freejoint.

### 4. Visualize
```bash
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
  --object_urdf models/dice/dice.urdf \
  --qpos_npz demo_results/g1/object_interaction/dice/Traj1_original.npz
```

## Verification
- **Frame alignment**: SMPL-X root `trans` ≈ CSV hip position → same world frame (no alignment needed).
- **Co-location**: dice center within ~1 hand-width of a wrist while manipulated (here: median 0.20 m).
- **Output**: `qpos.shape == (T, 43)`, no NaNs, cost decreasing/stable (here 0.84 → ~0.26).

## Gotchas
- **Frames must share one world.** If your SMPL-X solver re-centered the root, align it to the CSV
  (the CSV also holds the human skeleton — use it to recover the rigid transform, apply to the dice).
- **Dice center offset.** The pose uses the 3-corner centroid (~8 cm off the true cube center). The
  cube is symmetric so contact is fine; if it looks shifted in viser, pass
  `--center-offset x y z` (meters, body frame) to step 2.
- **Held object scaling.** Required `object_scale_about_floor` — without it the dice stays at its
  captured height (~1.1 m) while the robot hands drop to ~0.8 m, breaking the grasp.
- **Robot.** Only `g1`/`t1` are configured; H1 has no defaults/mapping. SMPL-X→robot mapping exists
  for `g1` only.
