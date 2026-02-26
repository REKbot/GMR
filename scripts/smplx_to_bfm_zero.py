import argparse
import pathlib
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


ROBOT_CHOICES = [
    "unitree_g1",
    "unitree_g1_with_hands",
    "unitree_h1",
    "unitree_h1_2",
    "booster_t1",
    "booster_t1_29dof",
    "stanford_toddy",
    "fourier_n1",
    "engineai_pm01",
    "kuavo_s45",
    "hightorque_hi",
    "galaxea_r1pro",
    "berkeley_humanoid_lite",
    "booster_k1",
    "pnd_adam_lite",
    "openloong",
    "tienkung",
    "fourier_gr3",
]


def _resolve_output_path(output, save_path, save_format):
    if output is not None and save_path is not None:
        raise ValueError("Use only one of --output or --save_path.")

    path_str = output if output is not None else save_path
    if path_str is None:
        raise ValueError("One of --output or --save_path is required.")

    out_path = pathlib.Path(path_str)
    if out_path.suffix.lower() in {".npz", ".pkl"}:
        return out_path

    if save_format == "auto":
        save_format = "pkl"

    if save_format not in {"npz", "pkl"}:
        raise ValueError("--save_format must be one of auto/npz/pkl")

    return out_path.with_suffix(f".{save_format}")


def _save_archive(path, motion_key, motion_dict):
    out_path = pathlib.Path(path)
    motions = {motion_key: motion_dict}

    if out_path.suffix.lower() == ".pkl":
        with out_path.open("wb") as f:
            pickle.dump(motions, f)
    elif out_path.suffix.lower() == ".npz":
        # Canonical BFM-Zero layout: single key containing a dict-of-motions object.
        np.savez(out_path, data=np.array(motions, dtype=object))
    else:
        raise ValueError("Output path extension must be .npz or .pkl")


def _retarget_smplx_to_qpos(smplx_file, robot, coord_correction):
    here = pathlib.Path(__file__).parent
    smplx_folder = here / ".." / "assets" / "body_models"

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, smplx_folder, coord_correction=coord_correction
    )

    tgt_fps = 30
    smplx_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
    )

    qpos_list = [retargeter.retarget(frame) for frame in smplx_frames]
    return np.asarray(qpos_list), float(aligned_fps)


def _qpos_to_bfm_motion(qpos, fps, robot):
    root_trans_offset = qpos[:, :3].astype(np.float32)

    # MuJoCo qpos root quaternion is [w, x, y, z]. Convert to [x, y, z, w].
    root_quat_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]

    dof = torch.from_numpy(qpos[:, 7:]).to(dtype=torch.float32)
    robot_kin = KinematicsModel(str(ROBOT_XML_DICT[robot]), device=torch.device("cpu"))

    # Local joint rotations for all non-root joints, with identity on fixed joints.
    joint_local_quat = robot_kin.dof_to_rot(dof).cpu().numpy()

    all_local_quat = np.concatenate([root_quat_xyzw[:, None, :], joint_local_quat], axis=1)
    pose_aa = R.from_quat(all_local_quat.reshape(-1, 4)).as_rotvec().reshape(
        all_local_quat.shape[0], all_local_quat.shape[1], 3
    )

    return {
        "root_trans_offset": root_trans_offset,
        "pose_aa": pose_aa.astype(np.float32),
        "fps": float(fps),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retarget SMPL-X stageii motion and export a BFM-Zero-compatible motion archive."
    )
    parser.add_argument("--smplx_file", type=str, required=True, help="Input unretargeted SMPL-X stageii .npz")
    parser.add_argument("--robot", choices=ROBOT_CHOICES, default="unitree_g1")
    parser.add_argument(
        "--coord_correction",
        choices=["auto", "none", "identity", "x+90", "x-90", "y+90", "y-90", "z+90", "z-90", "x180", "repo_default"],
        default="auto",
        help="Coordinate correction for alternate SMPL-X axis conventions.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output archive path (.npz or .pkl)")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Alias of --output (compatible with scripts/smplx_to_robot.py style).",
    )
    parser.add_argument(
        "--save_format",
        choices=["auto", "npz", "pkl"],
        default="auto",
        help="Only used when output path has no extension. auto defaults to pkl.",
    )
    parser.add_argument(
        "--motion_key",
        type=str,
        default=None,
        help="Key name of the motion inside archive. Defaults to output filename stem.",
    )

    args = parser.parse_args()

    output_path = _resolve_output_path(args.output, args.save_path, args.save_format)
    motion_key = args.motion_key or output_path.stem

    qpos, fps = _retarget_smplx_to_qpos(args.smplx_file, args.robot, args.coord_correction)
    motion = _qpos_to_bfm_motion(qpos, fps, args.robot)
    _save_archive(output_path, motion_key, motion)

    print(
        f"Saved {output_path} with motion key '{motion_key}', "
        f"root_trans_offset={motion['root_trans_offset'].shape}, "
        f"pose_aa={motion['pose_aa'].shape}, fps={motion['fps']}"
    )


if __name__ == "__main__":
    main()
