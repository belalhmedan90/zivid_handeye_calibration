from pathlib import Path
import numpy as np
import rospy

# Assuming these are in your python path / package
from zivid_handeye_calibration.zivid_iface import ZividInterface
from zivid_handeye_calibration.dataset_handler import DatasetManager
from zivid_handeye_calibration.ros_iface import get_tf_mat

np.set_printoptions(precision=3, suppress=True)

def main():
    print("\n" + "="*40)
    print("      ZIVID HAND-EYE CALIBRATION")
    print("="*40)
    print("[1] Capture samples (Live ROS + Zivid)")
    print("[2] Run calibration (Process existing dataset)")
    print("[q] Quit")

    choice = input("\nSelect option: ").strip().lower()

    if choice == "1":
        # --- CAPTURE SESSION ---
        rospy.init_node("zivid_capture_node")
        
        # Initialize Zivid and Dataset Manager
        # We use a default 'data' folder for the root of all datasets
        iface = ZividInterface(hand_to_eye=True) 
        manager = DatasetManager(base_dir="./data", create_folder=True)
        
        iface.configure_capture_assistant()
        
        print(f"\n📂 Session Directory: {manager.dataset_path}")
        print("Commands: [s] Capture Sample | [q] Finish and Exit")

        while not rospy.is_shutdown():
            cmd = input(f"Command (s/q) > ").strip().lower()

            if cmd == "s":
                # 1. Zivid Capture & Board Detection
                # We need the detection to confirm the sample is good before saving
                frame, detection = iface.capture_calibration_board()
                if detection is None:
                    print("❌ Board not detected. Reposition camera/robot and try again.")
                    continue
                
                # 2. TF Lookup (Robot Base to Flange)
                try:
                    # Note: Adjust these frame names to match your ROS TF tree
                    robot_pose = get_tf_mat("root_link", "link7")
                except Exception as e:
                    print(f"❌ TF Error: {e}")
                    continue

                # 3. Save Sample
                # We save the detection result into 'point_cloud.zdf' for later loading
                sample_path = manager.save_sample(
                    robot_pose=robot_pose,
                    frame=frame
                )
                
                # Double check: DatasetManager's save_sample should handle detection.save
                # If your specific DatasetManager doesn't, we'll force it here:
                # detection.save(str(sample_path / "point_cloud.zdf"))
                
                print(f"✅ Sample saved in: {sample_path.name}")

            elif cmd == "q":
                print("Capture session finalized.")
                break

    elif choice == "2":
        # --- CALIBRATION SESSION ---
        dataset_str = input("Enter Dataset Path: ").strip()
        ds_path = Path(dataset_str)

        if not ds_path.exists():
            print(f"❌ Path {ds_path} does not exist.")
            return

        # Initialize Manager and Zivid (no camera connection needed for offline calib)
        manager = DatasetManager(base_dir=str(ds_path.parent), create_folder=False)
        
        mode_input = input("Mode [eth (Eye-to-Hand) / eih (Eye-in-Hand)]: ").strip().lower()
        is_eth = (mode_input == "eth")
        iface = ZividInterface(hand_to_eye=is_eth)

        # 1. Load Samples & Re-detect Board
        # We MUST re-detect because Zivid cannot 'load' a DetectionResult directly
        print(f"🔄 Loading and detecting board in samples...")
        try:
            # This uses the fixed logic to load robot_pose.yaml and point_cloud.zdf
            raw_data = manager.load_dataset(str(ds_path))
        except Exception as e:
            print(f"❌ Load error: {e}")
            return

        if len(raw_data) < 3:
            print(f"⚠️ Error: Only found {len(raw_data)} valid samples. Zivid requires >= 3.")
            return

        # 2. Run the math
        print(f"🚀 Computing {mode_input.upper()} calibration...")
        try:
            # Passing the list of dicts: {'pose_matrix': np.array, 'detection': DetectionResult}
            result_matrix = iface.perform_hand_eye_calibration(raw_data)
            
            # 3. Save and Output
            # We save it into the dataset folder we just processed
            manager.save_final_results(
                result_matrix=result_matrix,
                output_dir=ds_path,
                camera_frame="zivid_optical_frame",
                reference_frame="root_link"
            )
            
            print("\n" + "="*20)
            print("CALIBRATION SUCCESS")
            print("="*20)
            print(result_matrix)
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")

    elif choice == "q":
        print("Exiting tool.")
    else:
        print("Invalid selection.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")