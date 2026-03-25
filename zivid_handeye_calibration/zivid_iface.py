from datetime import timedelta
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import zivid
import zivid.calibration


class ZividInterface:
    """
    A wrapper for the Zivid SDK to simplify camera connection, 
    calibration board detection, and hand-eye calibration.
    """

    def __init__(self, hand_to_eye: bool = True) -> None:
        """
        Initializes the Zivid Application and connects to the first available camera.
        
        Args:
            hand_to_eye:    If True, uses Eye-to-Hand (fixed camera). 
                            If False, uses Eye-in-Hand (camera on robot).
        """
        self.app = zivid.Application()
        self.camera: Optional[zivid.Camera] = None
        self.hand_to_eye = hand_to_eye
        self.settings: Optional[zivid.Settings] = None

    def _ensure_connected(self):
            """Internal helper to connect only when hardware is required."""
            if self.camera is None:
                print("Connecting to Zivid camera...")
                self.camera = self.app.connect_camera()

    def configure_capture_assistant(self, max_time_ms: int = 1200) -> None:
        """
        Uses Zivid Capture Assistant to automatically find optimal exposure settings.
        
        Args:
            max_time_ms: Maximum allowed capture time in milliseconds.
        """
        self._ensure_connected()
        suggest_params = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=timedelta(milliseconds=max_time_ms),
            ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.hz50
        )
        self.settings = zivid.capture_assistant.suggest_settings(
            self.camera, 
            suggest_params
        )

    def capture_calibration_board(self) -> Tuple[Optional[zivid.calibration.DetectionResult], zivid.Frame]:
        """
        Captures a frame and attempts to detect the Zivid calibration board.
        
        Returns:
            A DetectionResult object if successful, None otherwise.
        """
        self._ensure_connected()
        if not self.settings:
            self.configure_capture_assistant()

        frame = self.camera.capture(self.settings)
        detection = zivid.calibration.detect_calibration_board(frame)

        if not detection.valid():
            print("Warning: Calibration board not detected in the current frame.")
            return frame, None
            
        return frame, detection

    def perform_hand_eye_calibration(
        self, 
        samples: List[Dict[str, Union[np.ndarray, zivid.calibration.DetectionResult]]]
    ) -> np.ndarray:
        """
        Computes the 4x4 transformation matrix between the robot and the camera.
        
        Args:
            samples: A list of dicts, each containing:
                - "pose_matrix": 4x4 numpy array (Robot Base to Flange)
                - "detection": The result from capture_calibration_board()
        
        Returns:
            A 4x4 numpy array representing the calibration transformation.
        """
        calibration_inputs = []

        for s in samples:
            robot_pose = s["pose_matrix"]
            # Ensure we are using a 4x4 float64 matrix
            if not isinstance(robot_pose, np.ndarray) or robot_pose.shape != (4, 4):
                raise ValueError("Pose matrix must be a 4x4 numpy array.")

            # Convert numpy matrix to Zivid Pose
            z_pose = zivid.calibration.Pose(robot_pose)
            
            calibration_inputs.append(
                zivid.calibration.HandEyeInput(z_pose, s["detection"])
            )

        # Perform the actual math based on configuration
        if self.hand_to_eye:
            print("Performing Eye-to-Hand calibration...")
            result = zivid.calibration.calibrate_eye_to_hand(calibration_inputs)
        else:
            print("Performing Eye-in-Hand calibration...")
            result = zivid.calibration.calibrate_eye_in_hand(calibration_inputs)

        if not result.valid():
            raise RuntimeError("Hand-eye calibration failed: Result is invalid.")

        return np.array(result.transform())
