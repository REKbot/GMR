"""FBX mocap take -> world joint-position keypoints npz.

Output feeds scripts/retarget_t800_principled.py --keypoint_npz (same format as
the hulksmash pipeline): one (T,3) float64 array of world positions per LAFAN1
bone name, plus a 1-element "fps" array. Z-up right-handed (Blender world);
units are arbitrary (the retargeter's lift_scale normalizes them).

Runs inside Blender (headless), NOT the gmr conda env:

  blender --background --python scripts/fbx_to_keypoints.py -- \
      <take.fbx> <out.npz> [--armature NAME] [--start F] [--end F] \
      [--fps N] [--list-bones]

Bone matching: exact name first, then unique suffix match (handles prefixed
rigs like "mixamorig:LeftHand" or "Body:LeftHand"). Unresolvable keys print
all candidate bone names; use --list-bones to dump the rig and extend
KEY_ALIASES if a rig uses different names entirely.
"""

import argparse
import sys

import bpy
import numpy as np

# LAFAN1-style bone names retarget_t800_principled.load_keypoint_npz expects.
KEYS = [
    "Spine", "Spine1", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]

# Per-key alternative spellings seen in other rigs (extend as needed).
KEY_ALIASES = {
    "LeftToeBase": ["LeftToe"],
    "RightToeBase": ["RightToe"],
    "Spine1": ["Spine2"],  # any upper-spine bone works; retargeter reads it as chest
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="fbx_to_keypoints")
    ap.add_argument("fbx", help="input .fbx mocap take")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--armature", default=None,
                    help="armature object name if the FBX has several")
    ap.add_argument("--start", type=int, default=None,
                    help="first frame to export (default: action start)")
    ap.add_argument("--end", type=int, default=None,
                    help="last frame to export, inclusive (default: action end)")
    ap.add_argument("--fps", type=float, default=None,
                    help="override fps stamp (default: scene fps from the FBX)")
    ap.add_argument("--list-bones", action="store_true",
                    help="print the rig's bone names and exit")
    return ap.parse_args(argv)


def pick_armature(name):
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("[fbx_to_keypoints] no armature in FBX")
    if name is not None:
        match = [a for a in armatures if a.name == name]
        if not match:
            raise SystemExit("[fbx_to_keypoints] no armature named %r; have: %s"
                             % (name, [a.name for a in armatures]))
        return match[0]
    if len(armatures) > 1:
        print("[fbx_to_keypoints] multiple armatures, using first:",
              [a.name for a in armatures])
    return armatures[0]


def map_bones(arm):
    """Map each LAFAN1 key to a pose bone: exact name, then alias, then unique
    suffix match (prefixed rigs)."""
    bones = list(arm.pose.bones)
    by_name = {b.name: b for b in bones}
    out, unresolved = {}, {}
    for key in KEYS:
        candidates = [key] + KEY_ALIASES.get(key, [])
        hit = next((by_name[c] for c in candidates if c in by_name), None)
        if hit is None:
            for c in candidates:
                suffix = [b for b in bones if b.name.endswith(c)]
                if len(suffix) == 1:
                    hit = suffix[0]
                    break
                if len(suffix) > 1:
                    unresolved[key] = [b.name for b in suffix]
        if hit is not None:
            out[key] = hit
        elif key not in unresolved:
            unresolved[key] = []
    if unresolved:
        print("[fbx_to_keypoints] rig bones:", sorted(b.name for b in bones))
        raise SystemExit("[fbx_to_keypoints] unresolved keys (ambiguous or missing): %s"
                         % unresolved)
    return out


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.fbx)

    arm = pick_armature(args.armature)
    if args.list_bones:
        for b in arm.pose.bones:
            parent = b.parent.name if b.parent else None
            print("bone: %s  <- %s" % (b.name, parent))
        return

    scene = bpy.context.scene
    fps = args.fps if args.fps is not None else scene.render.fps / scene.render.fps_base
    ad = arm.animation_data
    if not (ad and ad.action):
        raise SystemExit("[fbx_to_keypoints] armature %r has no action" % arm.name)
    a0, a1 = (int(round(v)) for v in ad.action.frame_range)
    f0 = args.start if args.start is not None else a0
    f1 = args.end if args.end is not None else a1
    print("[fbx_to_keypoints] take: %s" % args.fbx)
    print("[fbx_to_keypoints] armature: %s  fps: %g  action: %d-%d  exporting: %d-%d"
          % (arm.name, fps, a0, a1, f0, f1))

    bone_map = map_bones(arm)
    renamed = {k: v.name for k, v in bone_map.items() if v.name != k}
    if renamed:
        print("[fbx_to_keypoints] non-exact bone matches:", renamed)

    n = f1 - f0 + 1
    data = {k: np.zeros((n, 3), dtype=np.float64) for k in KEYS}
    for i in range(n):
        scene.frame_set(f0 + i)
        mw = arm.matrix_world
        for k in KEYS:
            w = mw @ bone_map[k].head
            data[k][i] = (w.x, w.y, w.z)

    data["fps"] = np.array([fps], dtype=np.float64)
    np.savez(args.out, **data)
    hips = 0.5 * (data["LeftUpLeg"] + data["RightUpLeg"])
    travel = np.linalg.norm(hips[-1] - hips[0])
    print("[fbx_to_keypoints] saved %s  frames=%d fps=%g  hip-height f0=%.3f  hip travel=%.3f"
          % (args.out, n, fps, hips[0][2], travel))


if __name__ == "__main__":
    main()
