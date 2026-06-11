"""Principled BVH -> T800 retarget.

POSITION (location): each robot link tracks the human joint, laid out with the
ROBOT's own segment lengths (bone-direction retargeting) so proportions/stance
width are the robot's, not a scaled human's.

ORIENTATION (continuous, no flips): track the human bone WORLD QUATERNION with a
per-joint offset CALIBRATED ONCE by position-IK'ing the robot to match the human
at frame 0:
    offset_body = inv(R_human_bone(f0)) @ R_robot_body(qpos0)
    target(t)   = R_human_bone(t) @ offset_body
Quaternions interpolate continuously, so nothing flips over time. The offset is
the exact convention bridge at the real f0 correspondence (no guessing, no T-pose).

Tools:
  --view   robot + human reference skeleton (drawn beside it) in one window
  --diag   synthetic SYMMETRIC joint-test motion (sweeps each joint group)
  --check  prints robot-vs-human facing-angle metrics
Output pkl matches scripts/bvh_to_robot.py (root_rot stored XYZW).
"""

import argparse
import json
import math
import os
import pickle

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

IK_CONFIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    + "/general_motion_retargeting/ik_configs/bvh_lafan1_to_t800.json"

# Position chains: (human bones, robot bodies), root-first.
CHAINS = [
    (["Hips", "LeftUpLeg", "LeftLeg", "LeftFootMod"],
     ["LINK_BASE", "LINK_HIP_ROLL_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_ROLL_L"]),
    (["Hips", "RightUpLeg", "RightLeg", "RightFootMod"],
     ["LINK_BASE", "LINK_HIP_ROLL_R", "LINK_KNEE_PITCH_R", "LINK_ANKLE_ROLL_R"]),
    (["Hips", "Spine2"],
     ["LINK_BASE", "LINK_WAIST_YAW"]),
    (["Spine2", "LeftArm", "LeftForeArm", "LeftHand"],
     ["LINK_WAIST_YAW", "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_PITCH_L", "LINK_WRIST_END_L"]),
    (["Spine2", "RightArm", "RightForeArm", "RightHand"],
     ["LINK_WAIST_YAW", "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_PITCH_R", "LINK_WRIST_END_R"]),
]

# robot body : (human bone, position weight, orientation weight)
BODY = {
    "LINK_BASE": ("Hips", 50, 40),
    "LINK_HIP_ROLL_L": ("LeftUpLeg", 5, 35),
    "LINK_KNEE_PITCH_L": ("LeftLeg", 30, 30),
    "LINK_ANKLE_ROLL_L": ("LeftFootMod", 90, 25),
    "LINK_HIP_ROLL_R": ("RightUpLeg", 5, 35),
    "LINK_KNEE_PITCH_R": ("RightLeg", 30, 30),
    "LINK_ANKLE_ROLL_R": ("RightFootMod", 90, 25),
    "LINK_WAIST_YAW": ("Spine2", 30, 35),
    "LINK_SHOULDER_ROLL_L": ("LeftArm", 5, 35),
    "LINK_ELBOW_PITCH_L": ("LeftForeArm", 30, 30),
    "LINK_WRIST_END_L": ("LeftHand", 70, 25),
    "LINK_SHOULDER_ROLL_R": ("RightArm", 5, 35),
    "LINK_ELBOW_PITCH_R": ("RightForeArm", 30, 30),
    "LINK_WRIST_END_R": ("RightHand", 70, 25),
}
IDENTITY = [1.0, 0.0, 0.0, 0.0]

# Arm bodies whose orientation weight --arms_rot_weight scales. Their frame_pr
# reference is the shoulder line, which degenerates when the arm crosses the
# chest (hooks): the orientation target snaps and the incremental IK winds the
# shoulder into a limit-pinned branch (pitch at +2.79 instead of ~-1.9) that
# leaves the wrist short. Position targets alone cannot do that.
ARM_BODIES = {
    "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_PITCH_L", "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_PITCH_R", "LINK_WRIST_END_R",
}

# Human skeleton edges for drawing the reference (only bones the BVH provides).
HUMAN_EDGES = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Spine3"), ("Spine3", "Neck"), ("Neck", "Head"),
    ("Spine3", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine3", "RightShoulder"), ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
]


def P(fr, b):
    return np.asarray(fr[b][0])


def Q(fr, b):
    return np.asarray(fr[b][1])  # wxyz


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ---------- position-derived target frames (convention-free) ----------
# Each robot body's target world orientation is reconstructed from human joint
# POSITIONS only (limb direction + a stable reference axis), never from the BVH's
# per-joint Euler/quaternion conventions. The robot<->semantic-frame convention
# bridge is derived once from the robot's own neutral geometry, so there is no
# quaternion guessing and no sign flip (references are the pelvis/shoulder lines,
# which stay defined even when a knee/elbow straightens).

def frame_us(up, side):
    """Orthonormal world frame from an up axis and a side axis. Columns =
    [forward, side, up] (right-handed). Used for the pelvis and torso."""
    u = _norm(up)
    f = _norm(np.cross(side, u))
    s = np.cross(u, f)
    return np.column_stack([f, s, u])


def frame_pr(primary, ref):
    """Orthonormal world frame from a primary (down-the-bone) axis and a
    reference axis. Columns = [primary, secondary, tertiary] (right-handed).
    secondary = ref x primary (a hinge-axis proxy); falls back if ref || primary."""
    p = _norm(primary)
    s = np.cross(ref, p)
    if np.linalg.norm(s) < 1e-4:
        alt = np.array([0.0, 0.0, 1.0]) if abs(p[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        s = np.cross(alt, p)
    s = _norm(s)
    u = np.cross(p, s)
    return np.column_stack([p, s, u])


def body_frame(g, body):
    """Semantic world frame for a tracked robot body, built from positions
    looked up via g(body). Identical recipe for human (g -> human bone pos) and
    robot neutral (g -> robot body pos), so the convention bridge cancels."""
    # Trunk axis (mid-hip -> mid-shoulder): the body's NET lean, averaging out
    # lower-spine flexion. T800's torso is yaw-only, so the base must carry the
    # whole lean -- using the lower-spine segment over-reads it and hunches.
    trunk_up = (0.5 * (g("LINK_SHOULDER_ROLL_L") + g("LINK_SHOULDER_ROLL_R"))
                - 0.5 * (g("LINK_HIP_ROLL_L") + g("LINK_HIP_ROLL_R")))
    if body == "LINK_BASE":  # facing from the hip line
        return frame_us(trunk_up, g("LINK_HIP_ROLL_L") - g("LINK_HIP_ROLL_R"))
    if body == "LINK_WAIST_YAW":  # same up as base -> only yaw (shoulder line) is asked of J12
        return frame_us(trunk_up, g("LINK_SHOULDER_ROLL_L") - g("LINK_SHOULDER_ROLL_R"))
    pelvis_side = g("LINK_HIP_ROLL_L") - g("LINK_HIP_ROLL_R")
    if body == "LINK_HIP_ROLL_L":
        return frame_pr(g("LINK_KNEE_PITCH_L") - g("LINK_HIP_ROLL_L"), pelvis_side)
    if body == "LINK_HIP_ROLL_R":
        return frame_pr(g("LINK_KNEE_PITCH_R") - g("LINK_HIP_ROLL_R"), pelvis_side)
    if body == "LINK_KNEE_PITCH_L":
        return frame_pr(g("LINK_ANKLE_ROLL_L") - g("LINK_KNEE_PITCH_L"), pelvis_side)
    if body == "LINK_KNEE_PITCH_R":
        return frame_pr(g("LINK_ANKLE_ROLL_R") - g("LINK_KNEE_PITCH_R"), pelvis_side)
    if body == "LINK_ANKLE_ROLL_L":  # foot inherits shin direction (no toe body tracked)
        return frame_pr(g("LINK_ANKLE_ROLL_L") - g("LINK_KNEE_PITCH_L"), pelvis_side)
    if body == "LINK_ANKLE_ROLL_R":
        return frame_pr(g("LINK_ANKLE_ROLL_R") - g("LINK_KNEE_PITCH_R"), pelvis_side)
    sh_side = g("LINK_SHOULDER_ROLL_L") - g("LINK_SHOULDER_ROLL_R")
    if body == "LINK_SHOULDER_ROLL_L":
        return frame_pr(g("LINK_ELBOW_PITCH_L") - g("LINK_SHOULDER_ROLL_L"), sh_side)
    if body == "LINK_SHOULDER_ROLL_R":
        return frame_pr(g("LINK_ELBOW_PITCH_R") - g("LINK_SHOULDER_ROLL_R"), sh_side)
    # Forearm/hand reference the STABLE shoulder line, not the upper-arm direction:
    # at full punch extension the forearm goes collinear with the upper arm, which
    # would degenerate the elbow-plane cross product and snap (the apex flap).
    if body == "LINK_ELBOW_PITCH_L":
        return frame_pr(g("LINK_WRIST_END_L") - g("LINK_ELBOW_PITCH_L"), sh_side)
    if body == "LINK_ELBOW_PITCH_R":
        return frame_pr(g("LINK_WRIST_END_R") - g("LINK_ELBOW_PITCH_R"), sh_side)
    if body == "LINK_WRIST_END_L":  # hand inherits forearm direction
        return frame_pr(g("LINK_WRIST_END_L") - g("LINK_ELBOW_PITCH_L"), sh_side)
    if body == "LINK_WRIST_END_R":
        return frame_pr(g("LINK_WRIST_END_R") - g("LINK_ELBOW_PITCH_R"), sh_side)
    raise KeyError(body)


def bridges(rest_pos, rest_rot):
    """Per-body convention bridge C = F_robot_neutral^T @ R_robot_neutral, so that
    target_world(t) = F_human(t) @ C reproduces the robot body frame exactly at
    neutral and rotates rigidly with the human limb thereafter."""
    g = lambda body: rest_pos[body]
    return {body: body_frame(g, body).T @ rest_rot[body] for body in BODY}


def target_quats(fr, C):
    """World-orientation target (wxyz) per HUMAN BONE name for each tracked body."""
    g = lambda body: P(fr, BODY[body][0])
    out = {}
    for body in BODY:
        m = body_frame(g, body) @ C[body]
        out[BODY[body][0]] = R.from_matrix(m).as_quat(scalar_first=True)
    return out


def neutral_body(model):
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[0:3] = [0, 0, 1.03]
    data.qpos[3:7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)
    pos = {}
    for i in range(model.nbody):
        pos[mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)] = data.xpos[i].copy()
    return pos


def neutral_rot(model):
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[0:3] = [0, 0, 1.03]
    data.qpos[3:7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)
    rot = {}
    for i in range(model.nbody):
        rot[mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)] = data.xmat[i].reshape(3, 3).copy()
    return rot


def write_config(use_rot, arms_rot_scale=1.0):
    bones = sorted({w[0] for w in BODY.values()})
    table = {}
    for body, (bone, pw, rw) in BODY.items():
        rw_eff = rw * (arms_rot_scale if body in ARM_BODIES else 1.0)
        table[body] = [bone, pw, rw_eff if use_rot else 0, [0.0, 0.0, 0.0], list(IDENTITY)]
    config = {
        "robot_root_name": "LINK_BASE",
        "human_root_name": "Hips",
        "ground_height": 0.0,
        "human_height_assumption": 1.75,
        "use_ik_match_table1": True,
        "use_ik_match_table2": False,
        "human_scale_table": {b: 1.0 for b in bones},
        "ik_match_table1": table,
        "ik_match_table2": table,
    }
    with open(IK_CONFIG, "w") as f:
        json.dump(config, f, indent=4)


def seg_lengths(rest):
    out = []
    for bones, bodies in CHAINS:
        out.append([np.linalg.norm(rest[bodies[i + 1]] - rest[bodies[i]])
                    for i in range(len(bodies) - 1)])
    return out


def build_positions(fr, rest_seg, S):
    t = {"Hips": P(fr, "Hips") * S}
    for ci, (bones, bodies) in enumerate(CHAINS):
        anchor = t[bones[0]]
        for i in range(len(bones) - 1):
            d = _norm(P(fr, bones[i + 1]) - P(fr, bones[i]))
            if np.linalg.norm(d) < 1e-6:
                d = np.array([0.0, 0.0, -1.0])
            anchor = anchor + rest_seg[ci][i] * d
            t[bones[i + 1]] = anchor
    return t


def make_targets(fr, rest_seg, S, C):
    t = build_positions(fr, rest_seg, S)
    quats = target_quats(fr, C) if C is not None else None
    out = {}
    for b in t:
        out[b] = [t[b], quats[b] if quats is not None else np.array(IDENTITY)]
    return out


def retarget_frames(model, frames, rest_pos, rest_rot, rest_seg, S, arms_rot_scale=1.0):
    """Single pass: bone-direction POSITIONS + position-derived world-orientation
    targets. Orientations are reconstructed from human joint positions through the
    robot's own neutral-geometry convention bridge (no f0 IK, no quat conventions)."""
    C = bridges(rest_pos, rest_rot)
    write_config(use_rot=True, arms_rot_scale=arms_rot_scale)
    gmr = GMR(src_human="bvh_nokov", tgt_robot="t800", actual_human_height=1.75, verbose=False)
    return [gmr.retarget(make_targets(fr, rest_seg, S, C)).copy() for fr in frames]


def lift_scale(rest_seg, f0):
    human_leg = (np.linalg.norm(P(f0, "LeftLeg") - P(f0, "LeftUpLeg"))
                 + np.linalg.norm(P(f0, "LeftFootMod") - P(f0, "LeftLeg")))
    return (rest_seg[0][1] + rest_seg[0][2]) / human_leg


# ---------- diagnostic synthetic motion ----------

def diag_frames(n_per=40):
    """A symmetric joint-test built directly as robot qpos sweeps (no retarget):
    each joint group swings L and R together through its range and back. Lets you
    confirm joint directions/signs against the reference robot pose."""
    names = [
        ("HIP_PITCH", ["J00_HIP_PITCH_L", "J06_HIP_PITCH_R"]),
        ("HIP_ROLL", ["J01_HIP_ROLL_L", "J07_HIP_ROLL_R"]),
        ("HIP_YAW", ["J02_HIP_YAW_L", "J08_HIP_YAW_R"]),
        ("KNEE", ["J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R"]),
        ("ANKLE_PITCH", ["J04_ANKLE_PITCH_L", "J10_ANKLE_PITCH_R"]),
        ("SHOULDER_PITCH", ["J13_SHOULDER_PITCH_L", "J18_SHOULDER_PITCH_R"]),
        ("SHOULDER_ROLL", ["J14_SHOULDER_ROLL_L", "J19_SHOULDER_ROLL_R"]),
        ("SHOULDER_YAW", ["J15_SHOULDER_YAW_L", "J20_SHOULDER_YAW_R"]),
        ("ELBOW", ["J16_ELBOW_PITCH_L", "J21_ELBOW_PITCH_R"]),
        ("TORSO_YAW", ["J12_TORSO_YAW"]),
    ]
    return names, n_per


# ---------- views / renders ----------

def _add_sphere(scn, pos, rgba, size=0.025):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, [size, 0, 0], pos.astype(np.float64),
                    np.eye(3).flatten(), np.array(rgba, np.float32))
    scn.ngeom += 1


def _add_line(scn, a, b, rgba, w=0.012):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                    np.eye(3).flatten(), np.array(rgba, np.float32))
    mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, w, a.astype(np.float64), b.astype(np.float64))
    scn.ngeom += 1


def draw_human(scn, fr, S, offset):
    rgba = [0.95, 0.45, 0.1, 1.0]
    have = lambda n: n in fr
    for a, b in HUMAN_EDGES:
        if have(a) and have(b):
            _add_line(scn, P(fr, a) * S + offset, P(fr, b) * S + offset, rgba)
    for b in fr:
        if isinstance(fr[b][0], np.ndarray) or isinstance(fr[b][0], list):
            _add_sphere(scn, P(fr, b) * S + offset, rgba, 0.02)


def view_motion(model, qpos_list, frames, S, fps):
    import time
    import mujoco.viewer as mjv
    data = mj.MjData(model)
    # place the human reference 1.4 m to the robot's left (-y world)
    base0 = qpos_list[0][:3]
    href0 = (P(frames[0], "Hips") * S)
    offset = np.array([base0[0] - href0[0], base0[1] - href0[1] - 1.4, base0[2] - href0[2]])
    with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        i = 0
        while v.is_running():
            data.qpos[:] = qpos_list[i]
            mj.mj_forward(model, data)
            v.user_scn.ngeom = 0
            draw_human(v.user_scn, frames[i], S, offset)
            v.sync()
            time.sleep(1.0 / fps)
            i = (i + 1) % len(qpos_list)


def view_diag(model, fps):
    import time
    import mujoco.viewer as mjv
    groups, n_per = diag_frames()
    jadr = {}
    for i in range(model.njnt):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if nm:
            jadr[nm] = model.jnt_qposadr[i]
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[0:3] = [0, 0, 1.03]
    data.qpos[3:7] = [1, 0, 0, 0]
    seq = []
    for label, joints in groups:
        lo, hi = -0.8, 0.8
        for k in range(n_per):
            phase = math.sin(2 * math.pi * k / n_per)
            seq.append((label, joints, phase * hi))
    with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as v:
        i = 0
        while v.is_running():
            data.qpos[7:] = 0.0
            label, joints, val = seq[i]
            for j in joints:
                if j in jadr:
                    data.qpos[jadr[j]] = val
            mj.mj_forward(model, data)
            v.sync()
            time.sleep(1.0 / fps)
            i = (i + 1) % len(seq)


def _ang(a, b):
    a = a[:2] / (np.linalg.norm(a[:2]) + 1e-9)
    b = b[:2] / (np.linalg.norm(b[:2]) + 1e-9)
    return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def check(model, qpos_list, frames):
    data = mj.MjData(model)
    bid = lambda n: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
    print("frame | ROBOT base->foot base->waist base->punch | HUMAN pelvis->foot pelvis->punch")
    for fi in [400, 1028, 1542, 2000]:
        if fi >= len(qpos_list):
            continue
        data.qpos[:] = qpos_list[fi]; mj.mj_forward(model, data)
        bf = data.xmat[bid("LINK_BASE")].reshape(3, 3)[:, 0]
        rf = data.xmat[bid("LINK_ANKLE_ROLL_L")].reshape(3, 3)[:, 0]
        wf = data.xmat[bid("LINK_WAIST_YAW")].reshape(3, 3)[:, 0]
        punch = data.xpos[bid("LINK_WRIST_END_R")] - data.xpos[bid("LINK_SHOULDER_ROLL_R")]
        fr = frames[fi]
        hp = np.cross(P(fr, "LeftUpLeg") - P(fr, "RightUpLeg"), [0, 0, 1.0])
        hpunch = P(fr, "RightHand") - P(fr, "RightArm")
        hfoot = P(fr, "LeftToeBase") - P(fr, "LeftFoot")
        print(f"f{fi:5d}| {_ang(bf, rf):+5.0f} {_ang(bf, wf):+6.0f} {_ang(bf, punch):+7.0f}"
              f"  | {_ang(hp, hfoot):+5.0f} {_ang(hp, hpunch):+6.0f}")


def render(model, qpos_list, frames, S, render_dir, render_n):
    import imageio
    os.makedirs(render_dir, exist_ok=True)
    data = mj.MjData(model)
    ren = mj.Renderer(model, height=600, width=800)
    cam = mj.MjvCamera(); cam.distance = 3.4; cam.elevation = -12; cam.azimuth = 130
    base0 = qpos_list[0][:3]; href0 = P(frames[0], "Hips") * S
    offset = np.array([base0[0] - href0[0], base0[1] - href0[1] - 1.4, base0[2] - href0[2]])
    N = len(qpos_list)
    for k, fi in enumerate(np.linspace(0, N - 1, render_n).astype(int)):
        data.qpos[:] = qpos_list[fi]; mj.mj_forward(model, data)
        cam.lookat[:] = qpos_list[fi][:3]
        ren.update_scene(data, camera=cam)
        scn = ren.scene
        draw_human(scn, frames[fi], S, offset)
        imageio.imwrite(f"{render_dir}/frame_{k:02d}_f{fi}.png", ren.render())
    ren.close()
    print(f"rendered {render_n} frames to {render_dir}")


def video(model, qpos_list, frames, S, path, fps, stride=2):
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = mj.MjData(model)
    ren = mj.Renderer(model, height=720, width=1280)
    cam = mj.MjvCamera(); cam.distance = 3.8; cam.elevation = -6; cam.azimuth = 145
    base0 = qpos_list[0][:3]; href0 = P(frames[0], "Hips") * S
    offset = np.array([base0[0] - href0[0], base0[1] - href0[1] - 1.4, base0[2] - href0[2]])
    writer = imageio.get_writer(path, fps=fps // stride)
    for fi in range(0, len(qpos_list), stride):
        data.qpos[:] = qpos_list[fi]; mj.mj_forward(model, data)
        cam.lookat[:] = 0.5 * (qpos_list[fi][:3] + (P(frames[fi], "Hips") * S + offset))
        ren.update_scene(data, camera=cam)
        draw_human(ren.scene, frames[fi], S, offset)
        writer.append_data(ren.render())
    writer.close(); ren.close()
    print(f"wrote {path} ({len(qpos_list)//stride} frames)")


def load_keypoint_npz(path):
    """Load world joint POSITIONS (e.g. FK'd out of Blender from an FBX mocap) into
    the frames format the retarget consumes. Only positions are used (orientation is
    reconstructed from positions), so the source's bone-roll convention is irrelevant.
    Expected Z-up, right-handed; units are arbitrary (lift_scale normalizes them)."""
    d = np.load(path)
    fps = int(round(float(np.asarray(d["fps"]).reshape(-1)[0])))
    a = lambda k: np.asarray(d[k], dtype=np.float64)
    hips = 0.5 * (a("LeftUpLeg") + a("RightUpLeg"))
    src = {
        "Hips": hips, "Spine2": a("Spine1"),
        "LeftUpLeg": a("LeftUpLeg"), "LeftLeg": a("LeftLeg"), "LeftFootMod": a("LeftFoot"),
        "RightUpLeg": a("RightUpLeg"), "RightLeg": a("RightLeg"), "RightFootMod": a("RightFoot"),
        "LeftArm": a("LeftArm"), "LeftForeArm": a("LeftForeArm"), "LeftHand": a("LeftHand"),
        "RightArm": a("RightArm"), "RightForeArm": a("RightForeArm"), "RightHand": a("RightHand"),
        # extras for the reference-skeleton overlay
        "Spine": a("Spine"), "Spine1": a("Spine1"), "Neck": a("Neck"), "Head": a("Head"),
        "LeftShoulder": a("LeftShoulder"), "RightShoulder": a("RightShoulder"),
        "LeftFoot": a("LeftFoot"), "LeftToeBase": a("LeftToeBase"),
        "RightFoot": a("RightFoot"), "RightToeBase": a("RightToeBase"),
    }
    n = hips.shape[0]
    frames = [{k: [v[i], np.array(IDENTITY)] for k, v in src.items()} for i in range(n)]
    return frames, fps


def bvh_fps(path):
    """True clip fps from the BVH 'Frame Time' header. Frames are retargeted at the
    BVH's native rate (no decimation), so this is the correct fps to stamp. A wrong
    label (e.g. the old default 30 on a 120 fps capture) stretches the motion 4x and
    plays it in slow motion downstream."""
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("Frame Time:"):
                return int(round(1.0 / float(s.split(":")[1])))
    return 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_file")
    ap.add_argument("--keypoint_npz", default=None,
                    help="world joint-position npz (FK'd from FBX/Blender) instead of a BVH")
    ap.add_argument("--start_frame", type=int, default=None, help="trim: first source frame (inclusive)")
    ap.add_argument("--end_frame", type=int, default=None, help="trim: last source frame (exclusive)")
    ap.add_argument("--format", default="nokov", choices=["lafan1", "nokov"])
    ap.add_argument("--save_path")
    ap.add_argument("--motion_fps", type=int, default=None,
                    help="output fps label; default: auto-detected from the BVH Frame Time")
    ap.add_argument("--arms_rot_weight", type=float, default=1.0,
                    help="scale orientation weight on arm bodies (shoulder/elbow/wrist). "
                         "0 = position-only arms; use for cross-body moves (hooks) where the "
                         "orientation reference degenerates and flips the IK branch")
    ap.add_argument("--render_dir", default=None)
    ap.add_argument("--render_n", type=int, default=8)
    ap.add_argument("--video", default=None, help="write mp4 with human reference")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--diag", action="store_true", help="symmetric per-joint test (no bvh)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["t800"]))

    if args.diag:
        view_diag(model, args.motion_fps or 30)
        return

    rest = neutral_body(model)
    rest_rot = neutral_rot(model)
    rest_seg = seg_lengths(rest)
    if args.keypoint_npz:
        frames, kp_fps = load_keypoint_npz(args.keypoint_npz)
        fps = args.motion_fps or kp_fps
    else:
        frames, _ = load_bvh_file(args.bvh_file, format=args.format)
        fps = args.motion_fps or bvh_fps(args.bvh_file)
    if args.start_frame is not None or args.end_frame is not None:
        frames = frames[args.start_frame:args.end_frame]
        print(f"trimmed to source frames [{args.start_frame}:{args.end_frame}] -> {len(frames)} frames")
    S = lift_scale(rest_seg, frames[0])
    print(f"lift S={S:.3f}  motion_fps={fps}  frames={len(frames)}")

    qpos_list = retarget_frames(model, frames, rest, rest_rot, rest_seg, S,
                                arms_rot_scale=args.arms_rot_weight)

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])
        with open(args.save_path, "wb") as f:
            pickle.dump({"fps": fps, "root_pos": root_pos, "root_rot": root_rot,
                         "dof_pos": dof_pos, "local_body_pos": None, "link_body_list": None}, f)
        print(f"saved {args.save_path} frames={len(qpos_list)} dof={dof_pos.shape[1]} fps={fps}")
        print(f"  -> next: npy_to_npz.py --input_fps {fps} --fps 50  (NOT the default 30)")

    if args.check:
        check(model, qpos_list, frames)
    if args.render_dir:
        render(model, qpos_list, frames, S, args.render_dir, args.render_n)
    if args.video:
        video(model, qpos_list, frames, S, args.video, fps)
    if args.view:
        view_motion(model, qpos_list, frames, S, fps)


if __name__ == "__main__":
    main()
