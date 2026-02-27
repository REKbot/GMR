import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from rich import print
from tqdm import tqdm
import os
import numpy as np
import pickle
import sys

def _skeleton_motion_to_retarget_frames(motion):
    import torch
    from poselib.core.rotation3d import quat_mul_norm, quat_rotate

    global_positions = quat_rotate(
        torch.tensor([0.70711, 0, 0, 0.70711]),
        motion.global_translation,
    ).detach().cpu().numpy() / 100
    global_quaternions = quat_mul_norm(
        torch.tensor([0.70711, 0, 0, 0.70711]),
        motion.global_rotation,
    ).detach().cpu().numpy()

    joint_names = motion.skeleton_tree.node_names
    num_frames = global_positions.shape[0]
    num_joints = len(joint_names)
    data = []

    for frame in range(num_frames):
        motion_frame = {}
        for i in range(num_joints):
            motion_frame[joint_names[i].split("_")[1]] = [
                global_positions[frame, i].tolist(),
                global_quaternions[frame, i, [3, 0, 1, 2]].tolist(),
            ]
        data.append(motion_frame)

    return data




def _raise_fbx_dependency_error(motion_file, original_error):
    raise RuntimeError(
        "Failed to parse raw .fbx input. The Autodesk FBX Python SDK is required by "
        "third_party/poselib for FBX loading.\n"
        "Install FBX SDK Python bindings (module name `fbx`) and retry, or first convert "
        "the FBX to a PoseLib .pkl via `python third_party/poselib/fbx_importer.py ...` and "
        "pass that .pkl to --motion_file.\n"
        f"Input file: {motion_file}\n"
        f"Original error: {type(original_error).__name__}: {original_error}"
    ) from original_error

def load_optitrack_fbx_motion_file(motion_file, root_joint="Hips", fps=None):
    suffix = pathlib.Path(motion_file).suffix.lower()
    if suffix == ".pkl":
        with open(motion_file, "rb") as f:
            return pickle.load(f), None

    if suffix == ".fbx":
        print(
            f"Detected .fbx input. Parsing FBX directly with root_joint='{root_joint}', fps={fps}."
        )
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        third_party_path = repo_root / "third_party"
        if str(third_party_path) not in sys.path:
            sys.path.insert(0, str(third_party_path))

        # If another PoseLib distribution was imported earlier (e.g. pip `poselib`),
        # clear it so FBX parsing uses the repository-bundled implementation.
        for module_name in list(sys.modules.keys()):
            if module_name == "poselib" or module_name.startswith("poselib."):
                del sys.modules[module_name]

        try:
            from poselib.skeleton.skeleton3d import SkeletonMotion

            if "third_party/poselib" not in str(pathlib.Path(SkeletonMotion.__module__.replace('.', '/')).as_posix()):
                # Best-effort guard; the module file check below is definitive.
                pass
            poselib_module_path = pathlib.Path(sys.modules[SkeletonMotion.__module__].__file__).resolve()
            if "third_party/poselib" not in str(poselib_module_path):
                raise RuntimeError(
                    f"Unexpected PoseLib import path: {poselib_module_path}. "
                    "Expected repository-bundled PoseLib under third_party/poselib."
                )

            motion = SkeletonMotion.from_fbx(
                fbx_file_path=motion_file,
                root_joint=root_joint,
                fps=fps,
            )
            return _skeleton_motion_to_retarget_frames(motion), motion.fps
        except Exception as err:
            _raise_fbx_dependency_error(motion_file, err)

    raise ValueError(
        f"Unsupported motion file format '{suffix}'. Use a .pkl exported by PoseLib or a raw .fbx file."
    )


def _trim_frames_by_timestamp(data_frames, motion_fps, start_sec=None, end_sec=None):
    if start_sec is None and end_sec is None:
        return data_frames

    total_frames = len(data_frames)
    if total_frames == 0:
        return data_frames

    if motion_fps is None or motion_fps <= 0:
        raise ValueError(f"Invalid motion FPS for trimming: {motion_fps}")

    start_frame = 0 if start_sec is None else int(round(start_sec * motion_fps))
    end_frame = total_frames if end_sec is None else int(round(end_sec * motion_fps))

    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))

    print(
        f"Trimming input frames: {start_frame}–{end_frame} "
        f"({start_frame / motion_fps:.3f}s – {end_frame / motion_fps:.3f}s) "
        f"out of {total_frames} frames ({total_frames / motion_fps:.3f}s)"
    )
    return data_frames[start_frame:end_frame]

def offset_to_ground(retargeter: GMR, motion_data):
    offset = np.inf
    for human_data in motion_data:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(human_data, retargeter.human_root_name, retargeter.human_scale_table)
        human_data = retargeter.offset_human_data(human_data, retargeter.pos_offsets1, retargeter.rot_offsets1)
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]

    return offset

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_file",
        help="FBX motion file to load (OptiTrack motion).",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--root_joint",
        type=str,
        default="Hips",
        help="Root joint name when --motion_file is a raw .fbx.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional override FPS when --motion_file is a raw .fbx. If omitted, use FBX metadata.",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Optional start timestamp in seconds to trim input motion before retargeting.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Optional end timestamp in seconds to trim input motion before retargeting.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/optitrack_example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion (.pkl or .npz).",
    )
    
    
    args = parser.parse_args()

    if args.save_path is None:
        input_stem = pathlib.Path(args.motion_file).stem
        args.save_path = f"{input_stem}.pkl"
        print(f"No --save_path provided. Saving output to {args.save_path}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # Only create directory if it's not empty
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []

    
    # Load OptiTrack FMB motion trajectory
    print(f"Loading OptiTrack FBX motion file: {args.motion_file}")
    data_frames, detected_motion_fps = load_optitrack_fbx_motion_file(
        args.motion_file, root_joint=args.root_joint, fps=args.fps
    )
    motion_fps = detected_motion_fps if detected_motion_fps is not None else (args.fps or 120)
    data_frames = _trim_frames_by_timestamp(
        data_frames,
        motion_fps=motion_fps,
        start_sec=args.start,
        end_sec=args.end,
    )
    print(f"Loaded {len(data_frames)} frames")

    if len(data_frames) == 0:
        raise ValueError("No frames remain after trimming. Please adjust --start/--end.")

    # Initialize the retargeting system with fbx configuration
    retargeter = GMR(
        src_human="fbx_offline",  # Use the new fbx configuration
        tgt_robot=args.robot,
        actual_human_height=1.8,
    )

    height_offset = offset_to_ground(retargeter, data_frames)
    retargeter.set_ground_offset(height_offset)

    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=1,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            camera_follow=False,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps:.3f}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(data_frames), desc="Retargeting OptiTrack motion")
    
    # Start the viewer
    i = 0

    while i < len(data_frames):
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        qpos_list.append(qpos)

    import pickle
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # save from wxyz to xyzw
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    local_body_pos = None
    body_names = None

    motion_data = {
        "fps": motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }

    save_suffix = pathlib.Path(args.save_path).suffix.lower()
    if save_suffix == ".npz":
        np.savez(
            args.save_path,
            fps=np.array([motion_fps], dtype=np.float32),
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
        )
    else:
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
    print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close() 
