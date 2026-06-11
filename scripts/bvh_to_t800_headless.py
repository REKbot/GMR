"""Headless BVH -> T800 retarget. Same output pkl as scripts/bvh_to_robot.py
(root_rot stored XYZW), but no GUI viewer, and optional offscreen PNG frame dump
(EGL) for visual verification on machines without a display.

Run under MUJOCO_GL=egl for the frame dump.
"""

import argparse
import os
import pickle

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_file", required=True)
    ap.add_argument("--robot", default="t800")
    ap.add_argument("--format", default="nokov", choices=["lafan1", "nokov"])
    ap.add_argument("--save_path", required=True, help="output .pkl")
    ap.add_argument("--motion_fps", type=int, default=30)
    ap.add_argument("--render_dir", default=None, help="dir to dump preview PNGs")
    ap.add_argument("--render_n", type=int, default=6, help="num preview frames")
    args = ap.parse_args()

    frames, human_height = load_bvh_file(args.bvh_file, format=args.format)
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=human_height,
        verbose=False,
    )

    qpos_list = []
    for fr in frames:
        qpos_list.append(retargeter.retarget(fr).copy())

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([q[7:] for q in qpos_list])
    motion_data = {
        "fps": args.motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(
        f"[bvh_to_t800_headless] saved {args.save_path}  frames={len(qpos_list)} "
        f"dof={dof_pos.shape[1]} height={human_height}"
    )

    if args.render_dir is not None:
        import mujoco as mj

        os.makedirs(args.render_dir, exist_ok=True)
        model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        data = mj.MjData(model)
        renderer = mj.Renderer(model, height=600, width=800)
        cam = mj.MjvCamera()
        cam.distance = 3.0
        cam.elevation = -15
        cam.azimuth = 135
        n = len(qpos_list)
        idxs = np.linspace(0, n - 1, args.render_n).astype(int)
        for k, fi in enumerate(idxs):
            data.qpos[:] = qpos_list[fi]
            mj.mj_forward(model, data)
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            import imageio

            out = os.path.join(args.render_dir, f"frame_{k:02d}_f{fi}.png")
            imageio.imwrite(out, img)
            print(f"  rendered {out}")
        renderer.close()


if __name__ == "__main__":
    main()
