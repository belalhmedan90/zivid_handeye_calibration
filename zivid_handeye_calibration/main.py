from pathlib import Path
import numpy as np
import rclpy

# Assuming these are in your python path / package
from zivid_handeye_calibration.zivid_iface import ZividInterface
from zivid_handeye_calibration.dataset_handler import DatasetManager
# Updated imports for ROS 2 logic
from zivid_handeye_calibration.ros_iface import init_tf_handler, get_tf_mat

np.set_printoptions(precision=3, suppress=True)

def main():
    # Initialize rclpy at the very start
    if not rclpy.ok():
        rclpy.init()

    print("\n" + "="*40)
    print("      ZIVID HAND-EYE CALIBRATION (ROS 2)")
    print("="*40)
    print("[1] Capture samples (Live ROS 2 + Zivid)")
    print("[2] Run calibration (Process existing dataset)")
    print("[q] Quit")

    choice = input("\nSelect option: ").strip().lower()

    if choice == "1":
        # --- CAPTURE SESSION ---
        # Initialize our custom TF handler node
        init_tf_handler()
        
        # Initialize Zivid and Dataset Manager
        iface = ZividInterface(hand_to_eye=True) 
        manager = DatasetManager(base_dir="./data", create_folder=True)
        
        iface.configure_capture_assistant()
        
        print(f"\n📂 Session Directory: {manager.dataset_path}")
        print("Commands: [s] Capture Sample | [q] Finish and Exit")

        while rclpy.ok():
            # In ROS 2, if you want to ensure TF data is processing in the background 
            # while waiting for input, you might need a brief spin, 
            # but since get_tf_mat uses a timeout, it will handle the wait.
            cmd = input(f"Command (s/q) > ").strip().lower()

            if cmd == "s":
                # 1. Zivid Capture & Board Detection
                detection, frame = iface.capture_calibration_board()
                if detection is None:
                    print("❌ Board not detected. Reposition camera/robot and try again.")
                    continue
                
                # 2. TF Lookup (Robot Base to Flange)
                try:
                    # Adjust frame names to match your robot's URDF/TF tree
                    robot_pose = get_tf_mat("lara5_root_link", "lara5_flange")
                except Exception as e:
                    print(f"❌ TF Error: {e}")
                    continue

                # 3. Save Sample
                sample_path = manager.save_sample(
                    robot_pose=robot_pose,
                    frame=frame
                )
                
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

        manager = DatasetManager(base_dir=str(ds_path.parent), create_folder=False)
        
        mode_input = input("Mode [eth (Eye-to-Hand) / eih (Eye-in-Hand)]: ").strip().lower()
        is_eth = (mode_input == "eth")
        iface = ZividInterface(hand_to_eye=is_eth)

        print(f"🔄 Loading and detecting board in samples...")
        try:
            raw_data = manager.load_dataset(str(ds_path))
        except Exception as e:
            print(f"❌ Load error: {e}")
            return

        if len(raw_data) < 3:
            print(f"⚠️ Error: Only found {len(raw_data)} valid samples. Zivid requires >= 3.")
            return

        print(f"🚀 Computing {mode_input.upper()} calibration...")
        try:
            result_matrix = iface.perform_hand_eye_calibration(raw_data)
            
            manager.save_final_results(
                result_matrix=result_matrix,
                output_dir=ds_path,
                camera_frame="zivid_optical_frame",
                reference_frame="root_link"
            )
            
            result_matrix_m = result_matrix.copy()
            result_matrix_m[:3, 3] *= 0.001
            print("\n" + "="*20)
            print("CALIBRATION SUCCESS")
            print("="*20)
            print(result_matrix_m)
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")

    elif choice == "q":
        print("Exiting tool.")
    else:
        print("Invalid selection.")

    # Proper ROS 2 shutdown
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if rclpy.ok():
            rclpy.shutdown()
