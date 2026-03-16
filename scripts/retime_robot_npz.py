import argparse
import pathlib
import signal
import time

import numpy as np

from general_motion_retargeting import RobotMotionViewer


should_quit = False


def _request_quit(*_args):
    global should_quit
    should_quit = True


def keyboard_callback(keycode):
    """Quit preview with q / Q / ESC."""
    global should_quit
    if keycode in (27, ord("q"), ord("Q")):
        should_quit = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speed up or slow down a retargeted robot-motion NPZ by changing its playback FPS."
    )
    parser.add_argument(
        "--robot_motion_npz",
        type=str,
        required=True,
        help="Path to an input retargeted robot-motion npz file.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help="Playback speed factor. >1 speeds up, <1 slows down. Must be positive and not equal to 1.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Robot type used for preview.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop preview playback.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record preview video.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/retime_robot_npz_preview.mp4",
        help="Output path for preview video when --record_video is set.",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip viewer preview.",
    )
    return parser.parse_args()


def main():
    global should_quit
    args = parse_args()
    should_quit = False

    signal.signal(signal.SIGINT, _request_quit)
    signal.signal(signal.SIGTERM, _request_quit)

    if args.factor <= 0:
        raise ValueError("--factor must be positive.")
    if np.isclose(args.factor, 1.0):
        raise ValueError("--factor cannot be 1.0 (no speed change).")

    input_path = pathlib.Path(args.robot_motion_npz)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with np.load(input_path, allow_pickle=True) as motion_npz:
        motion_data = {k: motion_npz[k] for k in motion_npz.files}

    required_keys = ["fps", "root_pos", "root_rot", "dof_pos"]
    missing_keys = [k for k in required_keys if k not in motion_data]
    if missing_keys:
        raise KeyError(f"Missing required keys in npz: {missing_keys}")

    original_fps = float(motion_data["fps"])
    new_fps = original_fps * args.factor
    motion_data["fps"] = np.array(new_fps)

    speed_suffix = "faster" if args.factor > 1.0 else "slower"
    output_path = input_path.with_name(f"{input_path.stem}_{args.factor:g}_{speed_suffix}.npz")
    np.savez(output_path, **motion_data)

    print(f"Saved retimed motion to: {output_path}")
    print(f"Original FPS: {original_fps:g}")
    print(f"New FPS: {new_fps:g}")

    if args.no_preview:
        return

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"]
    dof_pos = motion_data["dof_pos"]

    # Stored motion uses root quaternion in xyzw order; viewer expects wxyz.
    root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=new_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
        keyboard_callback=keyboard_callback,
    )

    frame_idx = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    while viewer.viewer.is_running() and not should_quit:
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        viewer.step(
            root_pos=root_pos[frame_idx],
            root_rot=root_rot_wxyz[frame_idx],
            dof_pos=dof_pos[frame_idx],
            rate_limit=True,
            follow_camera=False,
        )

        frame_idx += 1
        if frame_idx >= len(root_pos):
            if args.loop:
                frame_idx = 0
            else:
                break

    viewer.close()


if __name__ == "__main__":
    main()
