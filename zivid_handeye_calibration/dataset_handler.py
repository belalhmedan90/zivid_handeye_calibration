from pathlib import Path
from datetime import datetime
import json
import yaml
import numpy as np
import cv2
import zivid
from scipy.spatial.transform import Rotation
from typing import List, Dict, Any


class DatasetManager:
    """
    Robust dataset manager for Zivid hand-eye calibration.
    """

    def __init__(self, base_dir: str, create_folder: bool = False):
        self.base_dir = Path(base_dir)

        if create_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_path: Path = self.base_dir / f"zivid_he_dataset_{timestamp}"
            self.dataset_path.mkdir(parents=True, exist_ok=True)
        else:
            self.dataset_path = self.base_dir

    # -------------------------------------------------------------------------
    # SAVE SAMPLE
    # -------------------------------------------------------------------------
    def save_sample(
        self,
        robot_pose: np.ndarray,
        frame: zivid.Frame,
    ) -> Path:
        """
        Saves:
        - robot_pose.yaml
        - point_cloud.zdf
        - color.png
        """

        # 🔴 HARD TYPE CHECK (prevents your exact bug)
        if not isinstance(frame, zivid.Frame):
            raise TypeError(
                f"Expected zivid.Frame, got {type(frame)}. "
                "Did you accidentally pass DetectionResult instead of frame?"
            )

        if robot_pose.shape != (4, 4):
            raise ValueError(f"robot_pose must be 4x4, got {robot_pose.shape}")

        # Create sample dir
        sample_idx = len(list(self.dataset_path.glob("sample_*")))
        sample_dir = self.dataset_path / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save pose
        pose_data = {
            "rows": 4,
            "cols": 4,
            "data": robot_pose.flatten().tolist()
        }

        with open(sample_dir / "robot_pose.yaml", "w", encoding="utf-8") as f:
            yaml.dump(pose_data, f)

        # 2. Save ZDF
        zdf_path = sample_dir / "point_cloud.zdf"
        print(f"💾 Saving ZDF: {zdf_path}")
        frame.save(str(zdf_path))

        # 3. Save RGB
        try:
            rgba = frame.point_cloud().copy_data("rgba")
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(str(sample_dir / "color.png"), bgr)
        except Exception as e:
            print(f"⚠️ Failed to save RGB image: {e}")

        return sample_dir

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    def validate_dataset(self, dataset_path: str):
        ds_path = Path(dataset_path)

        if not ds_path.exists():
            return False, "Path does not exist."

        samples = sorted(list(ds_path.glob("sample_*")))
        if not samples:
            return False, "No sample directories found."

        for s_dir in samples:
            files = {f.name for f in s_dir.iterdir()}

            if "point_cloud.zdf" not in files:
                return False, f"{s_dir.name}: missing point_cloud.zdf"

            if not ("robot_pose.yaml" in files or "robot_pose.npy" in files):
                return False, f"{s_dir.name}: missing robot_pose"

        return True, f"Dataset valid with {len(samples)} samples."

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        is_valid, msg = self.validate_dataset(dataset_path)
        if not is_valid:
            print(f"⚠️ Validation Warning: {msg}")

        ds_path = Path(dataset_path)
        dataset_content = []

        sample_dirs = sorted(list(ds_path.glob("sample_*")))
        print(f"🔍 Found {len(sample_dirs)} samples")

        for s_dir in sample_dirs:

            # --- Load pose ---
            pose = None

            pose_yaml = s_dir / "robot_pose.yaml"
            pose_npy = s_dir / "robot_pose.npy"

            if pose_npy.exists():
                pose = np.load(pose_npy)

            elif pose_yaml.exists():
                with open(pose_yaml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    pose = np.array(data["data"]).reshape(4, 4)

                    # ⚠️ Only scale if you KNOW your dataset is in meters
                    # pose[:3, 3] *= 1000

            if pose is None:
                print(f"⚠️ Skipping {s_dir.name}: no pose")
                continue

            pose = pose.astype(np.float64)

            # --- Load ZDF ---
            zdf_path = s_dir / "point_cloud.zdf"

            if not zdf_path.exists():
                print(f"⚠️ Skipping {s_dir.name}: no ZDF")
                continue

            try:
                frame = zivid.Frame(str(zdf_path))
                detection = zivid.calibration.detect_calibration_board(frame)

                if not detection.valid():
                    print(f"❌ {s_dir.name}: board NOT detected")
                    try:
                        print("   ↳", detection.status_description())
                    except Exception:
                        pass
                    continue

                dataset_content.append({
                    "sample_id": s_dir.name,
                    "pose_matrix": pose,
                    "detection": detection
                })

            except Exception as e:
                print(f"❌ {s_dir.name}: {e}")
                continue

        print(f"✅ Loaded {len(dataset_content)} valid samples")
        return dataset_content

    # -------------------------------------------------------------------------
    # SAVE RESULT
    # -------------------------------------------------------------------------
    def save_final_results(
        self,
        result_matrix,
        output_dir: Path,
        camera_frame="zivid_optical_frame",
        reference_frame="root_link"
    ):
        output_path = output_dir / "hand_eye_result.json"

        rot = Rotation.from_matrix(result_matrix[:3, :3])
        x, y, z, w = rot.as_quat()

        tx, ty, tz = result_matrix[:3, 3] / 1000.0

        output = {
            "camera_frame": camera_frame,
            "camera_name": "zivid",
            "date": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "pose": {
                "orientation": {"w": float(w), "x": float(x), "y": float(y), "z": float(z)},
                "position": {"x": float(tx), "y": float(ty), "z": float(tz)}
            },
            "reference_frame": reference_frame
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

        print(f"✅ Saved calibration result: {output_path}")