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
    A modular tool to save, load, and validate Hand-Eye datasets.
    Updated to handle Zivid-specific calibration input requirements.
    """

    def __init__(self, base_dir: str, create_folder: bool=False):
        self.base_dir = Path(base_dir)
        # Required for calibration; color/point_cloud are now optional
        self.required_files = {"robot_pose", "point_cloud.zdf"}

        if create_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_path: Path = self.base_dir / f"zivid_he_dataset_{timestamp}"
            self.dataset_path.mkdir(parents=True, exist_ok=True)

    def save_sample(
        self,
        robot_pose: np.ndarray,
        frame: zivid.Frame,
    ) -> Path:
        """
        Saves a single calibration sample:
        1. Robot pose as YAML
        2. Zivid point cloud (ZDF)
        3. RGB image for visual inspection
        """

        # Ensure robot_pose is 4x4
        if robot_pose.shape != (4, 4):
            raise ValueError(f"robot_pose must be 4x4, got {robot_pose.shape}")

        # Determine sample directory
        sample_idx = len(list(self.dataset_path.glob("sample_*")))
        sample_dir = self.dataset_path / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Robot Pose (YAML)
        pose_data = {"rows": 4, "cols": 4, "data": robot_pose.flatten().tolist()}
        with open(sample_dir / "robot_pose.yaml", "w", encoding="utf-8") as f:
            yaml.dump(pose_data, f)

        # 2. Save Zivid Point Cloud (ZDF)
        frame.save(str(sample_dir / "point_cloud.zdf"))

        # 3. Save RGB for visual inspection
        rgba = frame.point_cloud().copy_data("rgba")
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(sample_dir / "color.png"), bgr)

        return sample_dir

    def validate_dataset(self, dataset_path: str):
        """Checks if the folder structure contains at least the pose and detection."""
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            return False, "Path does not exist."

        samples = sorted(list(ds_path.glob("sample_*")))
        if not samples:
            return False, "No sample directories found."

        for s_dir in samples:
            existing_files = {f.name for f in s_dir.iterdir()}
            if "point_cloud.zdf" not in existing_files:
                return False, f"Missing point_cloud.zdf in {s_dir.name}"
            if not ("robot_pose.npy" in existing_files or "robot_pose.yaml" in existing_files):
                return False, f"Missing robot_pose in {s_dir.name}"

        return True, f"Dataset valid with {len(samples)} samples."

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
            """
            Iterates through the dataset, loads ZDF frames, and performs 
            board detection for the calibrator. 
            Supports robot poses in both .npy and .yaml formats.
            """
            is_valid, msg = self.validate_dataset(dataset_path)
            if not is_valid:
                print(f"⚠️ Validation Warning: {msg}")

            ds_path = Path(dataset_path)
            dataset_content = []

            # Find all sample directories
            sample_dirs = sorted(list(ds_path.glob("sample_*")))
            print(f"🔍 Found {len(sample_dirs)} sample folders in {ds_path.name}")

            for s_dir in sample_dirs:
                # 1. Load Robot Pose (NPY priority, YAML fallback)
                pose_path_npy = s_dir / "robot_pose.npy"
                pose_path_yaml = s_dir / "robot_pose.yaml"
                pose = None

                if pose_path_npy.exists():
                    pose = np.load(pose_path_npy)
                elif pose_path_yaml.exists():
                    with open(pose_path_yaml, "r", encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        # Handle your specific YAML structure: {"data": [...], "rows": 4, "cols": 4}
                        pose = np.array(data["data"]).reshape(4, 4)
                
                if pose is None:
                    print(f"Skipping {s_dir.name}: No robot_pose (.npy or .yaml) found.")
                    continue

                # Ensure the pose is a 4x4 float64 for Zivid SDK compatibility
                pose = pose.astype(np.float64)
                # print(f"{pose=}")
                # 2. Load the ZDF Frame (point_cloud.zdf)
                zdf_path = s_dir / "point_cloud.zdf"
                if not zdf_path.exists():
                    print(f"Skipping {s_dir.name}: point_cloud.zdf missing.")
                    continue

                # Load the Frame and detect the board
                # Note: This requires an active or initialized Zivid Application in some SDK versions
                try:
                    frame = zivid.Frame(str(zdf_path))
                    detection = zivid.calibration.detect_calibration_board(frame)

                    if not detection.valid():
                        print(f"❌ {s_dir.name}: Calibration board not found in point cloud.")
                        continue

                    # 3. Append to list
                    dataset_content.append({
                        "sample_id": s_dir.name,
                        "pose_matrix": pose,
                        "detection": detection
                    })
                except Exception as e:
                    print(f"❌ {s_dir.name}: Error processing ZDF: {e}")
                    continue

            print(f"✅ Successfully loaded {len(dataset_content)} valid samples.")
            return dataset_content

    def save_final_results(self, result_matrix, output_dir: Path, camera_frame="zivid_optical_frame", reference_frame="root_link"):
        """Saves final calibration result as JSON (meters)."""
        output_path = output_dir / "hand_eye_result.json"
        
        # Calibration usually returns Robot -> Camera. 
        # For a TF tree (Reference -> Camera), we often need the matrix as is or inverted
        # depending on EIH vs ETH. Here we provide the standard transform:
        
        rot_matrix = result_matrix[:3, :3]
        rot = Rotation.from_matrix(rot_matrix)
        x, y, z, w = rot.as_quat()

        # Convert translation from mm (Zivid default) to meters (ROS default)
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

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"\n✅ Calibration JSON saved to: {output_path}")