from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
import numpy as np


def load_robot_motion_any(motion_path):
    """Load robot motion from either legacy PKL or NPZ output."""
    ext = os.path.splitext(motion_path)[1].lower()

    if ext == ".npz":
        with np.load(motion_path, allow_pickle=True) as motion_npz:
            motion_data = {k: motion_npz[k] for k in motion_npz.files}

        required_keys = ["fps", "root_pos", "root_rot", "dof_pos"]
        missing_keys = [k for k in required_keys if k not in motion_data]
        if missing_keys:
            raise KeyError(f"Missing required keys in npz: {missing_keys}")

        fps_array = motion_data["fps"]
        motion_fps = float(fps_array[0]) if np.ndim(fps_array) > 0 else float(fps_array)

        motion_root_pos = motion_data["root_pos"]
        # Stored as xyzw; viewer expects wxyz.
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]
        motion_dof_pos = motion_data["dof_pos"]
        motion_local_body_pos = motion_data.get("local_body_pos")
        motion_link_body_list = motion_data.get("link_body_list")

        return (
            motion_data,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            motion_local_body_pos,
            motion_link_body_list,
        )

    return load_robot_motion(motion_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion_any(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()
