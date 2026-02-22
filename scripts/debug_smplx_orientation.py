import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def _canonical_fields(data):
    root_orient = data["root_orient"] if "root_orient" in data else data["global_orient"]
    trans = data["trans"] if "trans" in data else data["transl"]
    fps = data["mocap_frame_rate"] if "mocap_frame_rate" in data else data.get("fps", 30)
    return root_orient, trans, fps


def _candidates():
    def rx(deg):
        return R.from_euler("x", deg, degrees=True).as_matrix()

    def ry(deg):
        return R.from_euler("y", deg, degrees=True).as_matrix()

    def rz(deg):
        return R.from_euler("z", deg, degrees=True).as_matrix()

    return {
        "identity": np.eye(3),
        "x+90": rx(90),
        "x-90": rx(-90),
        "y+90": ry(90),
        "y-90": ry(-90),
        "z+90": rz(90),
        "z-90": rz(-90),
        "x180": rx(180),
        "repo_default": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    }


def score_candidate(rotvecs, trans, corr):
    world_up = np.array([0.0, 0.0, 1.0])

    root_rot = R.from_rotvec(rotvecs)
    corr_rot = R.from_matrix(corr)

    corrected = corr_rot * root_rot
    body_up = corrected.apply(world_up)

    # Upright: body up should align +Z.
    up_alignment = float(np.mean(body_up[:, 2]))

    # Motion plausibility: vertical translation should vary less than horizontal.
    corrected_trans = trans @ corr.T
    z_range = float(np.percentile(corrected_trans[:, 2], 95) - np.percentile(corrected_trans[:, 2], 5))
    xy_range = float(
        np.percentile(np.linalg.norm(corrected_trans[:, :2], axis=1), 95)
        - np.percentile(np.linalg.norm(corrected_trans[:, :2], axis=1), 5)
    )

    return {
        "up_alignment": up_alignment,
        "z_range": z_range,
        "xy_range": xy_range,
        "score": up_alignment - 0.2 * z_range,
    }


def main():
    parser = argparse.ArgumentParser(description="Debug likely SMPL-X world-axis correction.")
    parser.add_argument("--smplx_file", type=str, required=True)
    args = parser.parse_args()

    path = os.path.expanduser(args.smplx_file)
    data = np.load(path, allow_pickle=True)
    root_orient, trans, fps = _canonical_fields(data)

    print(f"Loaded: {path}")
    print(f"Keys: {list(data.files)}")
    print(f"root_orient shape: {root_orient.shape}, trans shape: {trans.shape}, fps: {fps}")

    results = []
    for name, corr in _candidates().items():
        m = score_candidate(root_orient, trans, corr)
        results.append((name, m))

    results.sort(key=lambda x: x[1]["score"], reverse=True)

    print("\nCandidate corrections (best first):")
    for name, m in results:
        print(
            f"- {name:12s} score={m['score']:+.3f} up_alignment={m['up_alignment']:+.3f} "
            f"z_range={m['z_range']:.3f} xy_range={m['xy_range']:.3f}"
        )

    best = results[0][0]
    print(f"\nBest candidate by heuristic: {best}")
    print("Use this only as a debugging hint; validate against the original video.")


if __name__ == "__main__":
    main()
