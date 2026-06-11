# Hand-off: Add T800 to GMR + retarget official boxing BVH (URKL competition)

Paste this into a fresh Claude Code context (ideally `cd d:\work\Rek\GMR`). It is self-contained.

## Mission

Add the **EngineAI T800** (25-DOF humanoid) as a target robot in **GMR** (General Motion
Retargeting, this repo), then retarget the official URKL **boxing BVH** clips to T800 motions in
the **NPY format** the downstream SONIC fine-tune consumes. This is the human->robot retargeting
step that earns **Item B** (raw-BVH self-remap) and **Item C** (self-collected mocap) in the
URKL preliminary submission.

Output of this task = T800 GMR config (committed) + retargeted `.npy` motions for
`straight_punch` and `540_spin_kick` (and later REK's own mocap), each `[base_pos(3),
base_quat wxyz(4), 25 joint_pos]` per frame in J00..J24 order.

## Why / where this fits (minimal context)

- Competition: URKL Global Humanoid Robot Free Fighting League (EngineAI). Deadline 2026-06-10.
- Strategy: fine-tune NVIDIA **SONIC** (GEAR-SONIC) on T800 to **overfit each demo move**, deploy
  in EngineAI's official MuJoCo sim. Retargeted motions feed: `npy -> npz -> motion_lib ->
  SONIC fine-tune -> export ONNX -> rl_sonic_example runner -> record`.
- Rubric Item B (10 pts) requires WE remap from raw **BVH** ourselves (not EngineAI's pre-made
  npy). GMR is the chosen retargeter (ICRA 2026, MIT, CPU real-time).
- Full project plan: `C:\Users\conta\.claude\plans\lets-work-together-and-synthetic-flurry.md`.
  Submission tree: `d:\work\Rek\URKL_submission\`. Pipeline doc:
  `d:\work\Rek\URKL_submission\workflow\motion_pipeline.md`.

## The clinching facts (use these)

- **Best template = `engineai_pm01`** (already in this repo): same vendor as T800,
  `robot_root_name: "LINK_BASE"`, EngineAI `LINK_*` body naming. Copy
  `general_motion_retargeting/ik_configs/bvh_lafan1_to_pm01.json` and `assets/engineai_pm01/`.
- **Official BVH is LAFAN1-style** (`ROOT Hips`, `Spine/Spine1.. Neck.. RightShoulder/RightArm/
  RightForeArm/RightHand`). Use the `bvh_lafan1_*` input path. `human_root_name: "Hips"`.
- GMR output per frame = `(base_translation, base_rotation [wxyz, MuJoCo], joint_positions)` in the
  robot MJCF joint order. Build the T800 mocap XML with joints in J00..J24 order so output is
  already in the SONIC order.

## T800 model facts

- **Source model:** `d:\work\Rek\urkl_official\t800_model\` (official): `xml/serial_t800.xml`
  (MJCF, with `xml/assets.xml`, `serial_links.xml`, `serial_actuators.xml`), `urdf/serial_t800.urdf`,
  `meshes/`, `texture/`. Mirror: `d:\work\Rek\rek-sim-trainer\robot_src\T800-new\`.
- **25 joints (order = J00..J24):** J00_HIP_PITCH_L, J01_HIP_ROLL_L, J02_HIP_YAW_L,
  J03_KNEE_PITCH_L, J04_ANKLE_PITCH_L, J05_ANKLE_ROLL_L, J06..J11 = right leg (same),
  J12_TORSO_YAW, J13_SHOULDER_PITCH_L, J14_SHOULDER_ROLL_L, J15_SHOULDER_YAW_L, J16_ELBOW_PITCH_L,
  J17_ELBOW_YAW_L, J18..J22 = right arm (SHOULDER_PITCH/ROLL/YAW, ELBOW_PITCH/YAW),
  J23_HEAD_PITCH, J24_HEAD_YAW.
- **Body link names:** LINK_BASE (root), LINK_HIP_PITCH/ROLL/YAW_{L,R}, LINK_KNEE_PITCH_{L,R},
  LINK_ANKLE_PITCH/ROLL_{L,R}, LINK_FOOT_{L,R}, **LINK_WAIST_YAW** (the torso/waist body — NOTE:
  the joint is named J12_TORSO_YAW but the LINK is LINK_WAIST_YAW), LINK_SHOULDER_PITCH/ROLL/YAW_{L,R},
  LINK_ELBOW_PITCH/YAW_{L,R}, LINK_WRIST_END_{L,R}, LINK_HEAD_PITCH, LINK_HEAD_YAW.
- Key bodies for the IK match table (map to human): pelvis->LINK_BASE, knees->LINK_KNEE_PITCH_{L,R},
  feet->LINK_ANKLE_ROLL_{L,R} (or LINK_FOOT_{L,R}), torso->LINK_WAIST_YAW,
  shoulders->LINK_SHOULDER_YAW_{L,R}, elbows->LINK_ELBOW_PITCH_{L,R}, hands/wrists->LINK_WRIST_END_{L,R},
  head->LINK_HEAD_YAW.

## Steps

1. **Env:**
   ```bash
   conda create -n gmr python=3.10 -y && conda activate gmr
   cd d:\work\Rek\GMR && pip install -e . && conda install -c conda-forge libstdcxx-ng -y
   ```
2. **T800 asset (mocap XML).** Create `assets/t800/t800_mocap.xml` modeled on
   `assets/engineai_pm01/*.xml`: a MuJoCo model of T800 with a free joint at LINK_BASE, the 25
   hinge joints in J00..J24 order, body frames named LINK_*, and visual geoms (meshes under
   `assets/t800/meshes/`, copied from `urkl_official/t800_model/meshes`). Easiest: adapt the
   official `serial_t800.xml` (resolve its `<include>`s / mesh paths) to GMR's mocap-xml shape.
3. **IK config.** Copy `ik_configs/bvh_lafan1_to_pm01.json` -> `ik_configs/bvh_lafan1_to_t800.json`
   and `smplx_to_pm01.json` -> `smplx_to_t800.json`. In each: keep `human_root_name: "Hips"`, set
   `robot_root_name: "LINK_BASE"`, and rewrite `ik_match_table1/2` robot body keys to T800 LINK_*
   names (see mapping above). Start from pm01's weights/offsets/`human_scale_table`; tune the
   per-body position/rotation offsets + scales for T800 proportions while viewing results.
4. **Register T800** in `general_motion_retargeting/params.py`:
   - `ROBOT_XML_DICT["t800"] = ASSET_ROOT / "t800" / "t800_mocap.xml"`
   - In `IK_CONFIG_DICT`, add `"t800"` under the smplx and bvh_lafan1 entries.
   - `ROBOT_BASE_DICT["t800"] = "LINK_BASE"`, plus `VIEWER_CAM_DISTANCE_DICT["t800"]`.
   - Add `"t800"` to the `--robot` choices in `scripts/bvh_to_robot.py` and `smplx_to_robot.py`.
5. **Retarget + visually verify:**
   ```bash
   python scripts/bvh_to_robot.py \
     --bvh_file d:/work/Rek/urkl_official/motions/bvh/straight_punch_zhiquan.bvh \
     --robot t800 --bvh_format lafan1 \
     --save_path d:/work/Rek/urkl_official/motions/retargeted/straight_punch_t800.pkl
   ```
   Use the MuJoCo viewer (RobotMotionViewer pops up) to confirm the punch looks right + balanced.
   Repeat for `540huixuantitui_001.bvh` (mapped/staged as `spin_kick_540.bvh`).
6. **pkl -> NPY (J00..J24).** Write `scripts/gmr_pkl_to_t800_npy.py` mapping the GMR pkl
   `(base_translation, base_rotation wxyz, joint_positions)` -> a `(T, 32)` float array:
   cols 0:3 = base_translation, 3:7 = base_rotation (wxyz), 7:32 = the 25 joints reordered to
   J00..J24. Confirm GMR's joint order vs J00..J24 (it follows the mocap XML order from step 2 —
   build that in J00..J24 to make this identity).
7. **Feed the SONIC pipeline** (downstream, mostly elsewhere):
   ```bash
   # patched converter (torso->waist fix already applied) + official URDF:
   python d:/work/Rek/urkl_official/training/whole_body_tracking_engineai/scripts/npy_to_npz.py \
     -i straight_punch_t800.npy -o straight_punch_50hz.npz --input_fps 30 --fps 50 --use_dfs \
     --urdf d:/work/Rek/urkl_official/t800_model/urdf/serial_t800.urdf
   # then (gear_sonic Isaac env): scripts/t800/npz_to_motion_lib.py -> motion_lib_t800/ -> finetune
   ```

## Verification
- GMR viewer: retargeted T800 motion looks like the human BVH, feet roughly planted, no wild jitter.
- `npy_to_npz.py` succeeds (prints `Frames .., Joints: 25, Bodies: 34`).
- Quaternions are wxyz (GMR + npy_to_npz both wxyz) — no extra conversion.

## Gotchas
- **LINK_WAIST_YAW** vs joint **J12_TORSO_YAW** naming mismatch (the official npy_to_npz had this
  bug; our copy is patched torso->waist).
- Foot/wrist leaf bodies: GMR IK targets a real body (use LINK_ANKLE_ROLL / LINK_WRIST_END).
- The official boxing BVH includes finger joints (RightHandIndex.. etc.) — T800 has no fingers;
  ignore them in the IK match table.
- Confirm `--bvh_format` flag name in `scripts/bvh_to_robot.py` (choices were `lafan1`/`nokov`).

## Deliverables
1. T800 support committed to REKbot/GMR (assets/t800, ik_configs/*_to_t800.json, params.py, CLI).
2. Retargeted `straight_punch_t800.npy` + `spin_kick_540_t800.npy` in
   `d:\work\Rek\urkl_official\motions\retargeted\`.
3. Workflow written up for Item B in `d:\work\Rek\URKL_submission\workflow\bvh_to_t800_retargeting.md`.
4. (Then) same flow for a REK self-collected fight clip -> Item C.
