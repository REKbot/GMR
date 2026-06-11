"""Generate bvh_lafan1_to_t800.json from real geometry instead of hand-tuned
pm01 numbers. Strategy:

- Position-driven limb matching (robot body ORIGIN tracks human joint position).
  Position targets are convention-independent, so they cannot pretzel the way
  orientation targets with a wrong offset do.
- Per-limb scale computed from constant BVH segment lengths vs robot segment
  lengths, so the scaled human limbs match the robot's reach (no crouch/overreach).
- Base + foot ORIENTATION offsets computed from the robot neutral pose and the
  human reference frame, so global facing and foot-flatness are tracked without
  guessing quaternions.

Run: python scripts/build_t800_ik_config.py
"""

import json
import pathlib

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "general_motion_retargeting" / "ik_configs" / "bvh_lafan1_to_t800.json"
REF_BVH = "/mnt/d/work/Rek/urkl_official/motions/bvh/straight_punch_zhiquan.bvh"

# robot body : human bone
PAIRS = {
    "LINK_BASE": "Hips",
    "LINK_HIP_ROLL_L": "LeftUpLeg",
    "LINK_KNEE_PITCH_L": "LeftLeg",
    "LINK_ANKLE_ROLL_L": "LeftFootMod",
    "LINK_HIP_ROLL_R": "RightUpLeg",
    "LINK_KNEE_PITCH_R": "RightLeg",
    "LINK_ANKLE_ROLL_R": "RightFootMod",
    "LINK_WAIST_YAW": "Spine2",
    "LINK_SHOULDER_ROLL_L": "LeftArm",
    "LINK_ELBOW_PITCH_L": "LeftForeArm",
    "LINK_WRIST_END_L": "LeftHand",
    "LINK_SHOULDER_ROLL_R": "RightArm",
    "LINK_ELBOW_PITCH_R": "RightForeArm",
    "LINK_WRIST_END_R": "RightHand",
}

# (pos_weight, rot_weight) per robot body. Orientation-DRIVEN limbs (so the scale
# applied to positions cannot splay the legs); position anchors only the base and
# feet for global placement + foot contact. Every body gets a geometry-derived
# rotation offset (robot neutral <-> human reference frame).
WEIGHTS = {
    "LINK_BASE": (60, 15),
    # Legs: orientation-driven so the scaled stance width cannot splay them.
    "LINK_HIP_ROLL_L": (0, 10),
    "LINK_KNEE_PITCH_L": (0, 14),
    "LINK_ANKLE_ROLL_L": (60, 18),
    "LINK_HIP_ROLL_R": (0, 10),
    "LINK_KNEE_PITCH_R": (0, 14),
    "LINK_ANKLE_ROLL_R": (60, 18),
    "LINK_WAIST_YAW": (0, 10),
    # Arms: position (hand LOCATION -> guard/punch reach) + orientation (punch
    # AXIS / forearm direction).
    "LINK_SHOULDER_ROLL_L": (35, 8),
    "LINK_ELBOW_PITCH_L": (45, 10),
    "LINK_WRIST_END_L": (70, 6),
    "LINK_SHOULDER_ROLL_R": (35, 8),
    "LINK_ELBOW_PITCH_R": (45, 10),
    "LINK_WRIST_END_R": (70, 6),
}

IDENTITY = [1.0, 0.0, 0.0, 0.0]


def body_world(model, data, name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy(), data.xquat[bid].copy()  # pos, quat wxyz


def main():
    # --- robot neutral pose ---
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["t800"]))
    data = mj.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[0:3] = [0, 0, 1.03]
    data.qpos[3:7] = [1, 0, 0, 0]
    mj.mj_forward(model, data)
    rob = {b: body_world(model, data, b) for b in PAIRS}

    # --- human reference frame (frame 0) ---
    frames, _ = load_bvh_file(REF_BVH, format="nokov")
    f0 = frames[0]
    hum = {bone: (np.asarray(f0[bone][0]), np.asarray(f0[bone][1])) for bone in PAIRS.values()}

    root_r = rob["LINK_BASE"][0]
    root_h = hum["Hips"][0]

    def seg(robot_bodies, human_bones):
        rl = sum(
            np.linalg.norm(rob[robot_bodies[i + 1]][0] - rob[robot_bodies[i]][0])
            for i in range(len(robot_bodies) - 1)
        )
        hl = sum(
            np.linalg.norm(hum[human_bones[i + 1]][0] - hum[human_bones[i]][0])
            for i in range(len(human_bones) - 1)
        )
        return rl / hl

    leg_scale = seg(
        ["LINK_HIP_ROLL_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_ROLL_L"],
        ["LeftUpLeg", "LeftLeg", "LeftFootMod"],
    )
    arm_scale = seg(
        ["LINK_SHOULDER_ROLL_L", "LINK_ELBOW_PITCH_L", "LINK_WRIST_END_L"],
        ["LeftArm", "LeftForeArm", "LeftHand"],
    )
    # torso scale: hips->shoulder vertical span
    torso_scale = np.linalg.norm(
        rob["LINK_SHOULDER_ROLL_L"][0] - root_r
    ) / np.linalg.norm(hum["LeftArm"][0] - root_h)

    print(f"leg_scale={leg_scale:.3f} arm_scale={arm_scale:.3f} torso_scale={torso_scale:.3f}")

    # Root scales by the leg factor so the base rides at the right height above
    # the (also leg-scaled) feet -- effectively a uniform lift of the whole human.
    scale_table = {
        "Hips": float(leg_scale),
        "Spine2": float(torso_scale),
        "LeftUpLeg": float(leg_scale), "RightUpLeg": float(leg_scale),
        "LeftLeg": float(leg_scale), "RightLeg": float(leg_scale),
        "LeftFootMod": float(leg_scale), "RightFootMod": float(leg_scale),
        "LeftArm": float(arm_scale), "RightArm": float(arm_scale),
        "LeftForeArm": float(arm_scale), "RightForeArm": float(arm_scale),
        "LeftHand": float(arm_scale), "RightHand": float(arm_scale),
    }

    # orientation offsets where we want them: base + feet.
    def offset(robot_body, human_bone):
        r_rob = R.from_quat(rob[robot_body][1], scalar_first=True)
        r_hum = R.from_quat(hum[human_bone][1], scalar_first=True)
        return (r_hum.inv() * r_rob).as_quat(scalar_first=True).tolist()

    # Geometry-derived rotation offset for EVERY matched body: at the reference
    # frame the robot body sits at its neutral orientation; thereafter it tracks
    # the human bone's world rotation. Requires the human reference frame to be a
    # roughly-neutral standing pose (frame 0 of the punch clip is).
    rot_offsets = {b: offset(b, bone) for b, bone in PAIRS.items()}

    table = {}
    for robot_body, human_bone in PAIRS.items():
        pw, rw = WEIGHTS[robot_body]
        roff = rot_offsets.get(robot_body, IDENTITY)
        table[robot_body] = [human_bone, pw, rw, [0.0, 0.0, 0.0], roff]

    config = {
        "robot_root_name": "LINK_BASE",
        "human_root_name": "Hips",
        "ground_height": 0.0,
        "human_height_assumption": 1.75,  # loader returns 1.75 -> ratio 1.0
        "use_ik_match_table1": True,
        "use_ik_match_table2": False,
        "human_scale_table": scale_table,
        "ik_match_table1": table,
        "ik_match_table2": table,
    }

    with open(OUT, "w") as f:
        json.dump(config, f, indent=4)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
