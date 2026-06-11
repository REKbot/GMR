"""Convert a GMR retargeting pkl (from bvh_to_robot.py / smplx_to_robot.py) into
the T800 SONIC .npy format consumed downstream by npy_to_npz.py.

GMR pkl layout (see scripts/bvh_to_robot.py save block):
    fps         : int
    root_pos    : (T, 3)            base translation, meters, MuJoCo world frame
    root_rot    : (T, 4)            base rotation, XYZW  (GMR saves wxyz->xyzw on disk)
    dof_pos     : (T, 25)           joint positions in MuJoCo model joint order

T800 npy layout: (T, 32) float32, per frame:
    cols  0:3   base translation
    cols  3:7   base rotation, WXYZ  (npy_to_npz expects wxyz, no extra convert)
    cols  7:32  25 joints in J00..J24 order

The T800 mocap XML defines its joints depth-first in J00..J24 order, so the
GMR dof order already equals J00..J24 and the joint copy is an identity. The
canonical order is asserted below; pass --xml to re-derive and reorder from the
live MuJoCo model if you ever change the body tree.
"""

import argparse
import pickle

import numpy as np

# Canonical SONIC joint order for T800 (J00..J24).
T800_JOINT_ORDER = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH", "J24_HEAD_YAW",
]


def gmr_dof_order(xml_path):
    """Return the actuated joint names in MuJoCo qpos order for the given model."""
    import os

    import mujoco as mj

    model = mj.MjModel.from_xml_path(os.path.abspath(xml_path))
    names = []
    for i in range(model.nv):
        jid = model.dof_jntid[i]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if name not in names:
            names.append(name)
    # Drop the free joint (root) which has no name or is the base.
    return [n for n in names if n in T800_JOINT_ORDER]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="GMR pkl path")
    parser.add_argument("-o", "--output", required=True, help="output .npy path")
    parser.add_argument(
        "--xml",
        default=None,
        help="optional T800 mocap XML; reorders dof to J00..J24 from the live model",
    )
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)  # (T, 3)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float64)  # (T, 4) xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)  # (T, 25)

    n_frames = root_pos.shape[0]
    assert root_rot_xyzw.shape == (n_frames, 4), root_rot_xyzw.shape
    assert dof_pos.shape[1] == len(T800_JOINT_ORDER), (
        f"expected {len(T800_JOINT_ORDER)} joints, got {dof_pos.shape[1]}"
    )

    # XYZW (pkl on disk) -> WXYZ (npy / npy_to_npz expectation).
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    # Reorder joints into canonical J00..J24 if a model is supplied; otherwise the
    # GMR order already equals J00..J24 (the mocap XML is authored in that order).
    if args.xml is not None:
        src_order = gmr_dof_order(args.xml)
        assert len(src_order) == len(T800_JOINT_ORDER), (src_order, len(src_order))
        index = [src_order.index(j) for j in T800_JOINT_ORDER]
        if index != list(range(len(index))):
            print(f"[gmr_pkl_to_t800_npy] reordering dof: model order != J00..J24")
        dof_pos = dof_pos[:, index]

    out = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1).astype(np.float32)
    assert out.shape == (n_frames, 32), out.shape

    np.save(args.output, out)
    print(
        f"[gmr_pkl_to_t800_npy] {args.input} -> {args.output}  "
        f"frames={n_frames} fps={data.get('fps')} shape={out.shape} (3 pos + 4 quat wxyz + 25 joints)"
    )


if __name__ == "__main__":
    main()
