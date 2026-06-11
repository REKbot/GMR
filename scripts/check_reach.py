"""Diagnose retarget reach: FK the saved pkl qpos and compare each tracked
body's world position against the position TARGET built from the keypoints,
frame by frame. Reports worst-residual frames per body and which joints sit
at their limits there. Use after retarget_t800_principled.py when a limb
visibly misses the mocap (e.g. a hook coming in short).

  python scripts/check_reach.py --keypoint_npz kp.npz --pkl out.pkl \
      [--start_frame N --end_frame M] [--bodies LINK_WRIST_END_L,...]
"""

import argparse
import pickle

import numpy as np
import mujoco as mj

import retarget_t800_principled as rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint_npz", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--start_frame", type=int, default=None)
    ap.add_argument("--end_frame", type=int, default=None)
    ap.add_argument("--bodies", default="LINK_WRIST_END_L,LINK_WRIST_END_R,LINK_ANKLE_ROLL_L,LINK_ANKLE_ROLL_R")
    ap.add_argument("--top", type=int, default=8, help="worst frames to print per body")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(str(rp.ROBOT_XML_DICT["t800"]))
    rest = rp.neutral_body(model)
    rest_seg = rp.seg_lengths(rest)

    frames, _ = rp.load_keypoint_npz(args.keypoint_npz)
    frames = frames[args.start_frame:args.end_frame]
    S = rp.lift_scale(rest_seg, frames[0])

    with open(args.pkl, "rb") as f:
        p = pickle.load(f)
    root_pos, root_rot, dof = p["root_pos"], p["root_rot"], p["dof_pos"]
    T = dof.shape[0]
    assert T == len(frames), f"pkl frames {T} != trimmed keypoint frames {len(frames)}"

    bodies = args.bodies.split(",")
    bid = {b: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies}
    bone_of = {b: rp.BODY[b][0] for b in bodies}

    jnames, jadr, jlo, jhi = [], [], [], []
    for i in range(model.njnt):
        if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE:
            jnames.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i))
            jadr.append(model.jnt_qposadr[i])
            jlo.append(model.jnt_range[i][0])
            jhi.append(model.jnt_range[i][1])
    jadr, jlo, jhi = np.array(jadr), np.array(jlo), np.array(jhi)

    data = mj.MjData(model)
    res = {b: np.zeros(T) for b in bodies}
    qall = np.zeros((T, len(jadr)))
    for t in range(T):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_rot[t][[3, 0, 1, 2]]  # xyzw -> wxyz
        data.qpos[7:] = dof[t]
        mj.mj_forward(model, data)
        targets = rp.build_positions(frames[t], rest_seg, S)
        for b in bodies:
            res[b][t] = np.linalg.norm(data.xpos[bid[b]] - targets[bone_of[b]])
        qall[t] = data.qpos[jadr]

    margin = 0.02  # rad from a limit counts as saturated
    for b in bodies:
        worst = np.argsort(res[b])[::-1][:args.top]
        print(f"\n=== {b} (target bone {bone_of[b]})  mean={res[b].mean():.3f} m  max={res[b].max():.3f} m")
        for t in sorted(worst):
            at_lim = [jnames[k] for k in range(len(jadr))
                      if qall[t, k] < jlo[k] + margin or qall[t, k] > jhi[k] - margin]
            print(f"  frame {t:4d}: residual {res[b][t]:.3f} m   at-limit: {at_lim}")


if __name__ == "__main__":
    main()
