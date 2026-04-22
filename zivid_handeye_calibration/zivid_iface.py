from datetime import timedelta
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import zivid
import zivid.calibration


class ZividInterface:
    """
    Robust Zivid wrapper for:
    - Camera connection
    - Capture configuration
    - Calibration board detection
    - Hand-eye calibration

    Designed to work reliably on newer SDKs (e.g., Ubuntu 24.04).
    """

    def __init__(self, hand_to_eye: bool = True, use_capture_assistant: bool = False) -> None:
        self.app = zivid.Application()
        self.camera: Optional[zivid.Camera] = None
        self.hand_to_eye = hand_to_eye
        self.settings: Optional[zivid.Settings] = None
        self.use_capture_assistant = use_capture_assistant

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    def _ensure_connected(self):
        if self.camera is None:
            print("🔌 Connecting to Zivid camera...")
            self.camera = self.app.connect_camera()
            print(f"✅ Connected to: {self.camera.info.serial_number}")

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    def configure_capture_assistant(self, max_time_ms: int = 2000) -> None:
        """Use Capture Assistant (can fail on newer SDKs depending on lighting)."""
        self._ensure_connected()

        print("⚙️ Using Capture Assistant...")
        suggest_params = zivid.capture_assistant.SuggestSettingsParameters(
            max_capture_time=timedelta(milliseconds=max_time_ms),
            ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.hz50,
        )

        self.settings = zivid.capture_assistant.suggest_settings(
            self.camera,
            suggest_params
        )

    def configure_manual(self) -> None:
        """
        Manual fallback settings (recommended if detection fails).
        These are intentionally conservative and stable.
        """
        print("⚙️ Using MANUAL capture settings (robust fallback)...")

        settings = zivid.Settings()

        acq = zivid.Settings.Acquisition()
        acq.aperture = 5.66
        acq.exposure_time = timedelta(milliseconds=10)
        acq.gain = 1.0
        acq.brightness = 1.0

        settings.acquisitions.append(acq)
        self.settings = settings

    def _ensure_settings(self):
        if self.settings is None:
            if self.use_capture_assistant:
                self.configure_capture_assistant()
            else:
                self.configure_manual()

    # -------------------------------------------------------------------------
    # Capture + Detection
    # -------------------------------------------------------------------------
    def capture_calibration_board(
        self,
        save_debug_frame: Optional[str] = None
    ) -> Tuple[Optional[zivid.calibration.DetectionResult], zivid.Frame]:
        """
        Capture frame and detect calibration board.

        Returns:
            (detection, frame)
        """
        self._ensure_connected()
        self._ensure_settings()

        print("📸 Capturing frame...")
        frame = self.camera.capture(self.settings)

        if save_debug_frame:
            print(f"💾 Saving debug frame: {save_debug_frame}")
            frame.save(save_debug_frame)

        print("🔍 Detecting calibration board...")
        detection = zivid.calibration.detect_calibration_board(frame)

        if not detection.valid():
            print("❌ Calibration board NOT detected!")
            try:
                print("ℹ️ Detection info:", detection.status_description())
            except Exception:
                print("ℹ️ No detailed status available from SDK.")
            return None, frame

        print("✅ Calibration board detected.")
        return detection, frame

    # -------------------------------------------------------------------------
    # Hand-Eye Calibration
    # -------------------------------------------------------------------------
    def perform_hand_eye_calibration(
        self,
        samples: List[Dict[str, Union[np.ndarray, zivid.calibration.DetectionResult]]]
    ) -> np.ndarray:
        """
        Perform hand-eye calibration.

        Each sample must contain:
            - "pose_matrix": 4x4 np.ndarray
            - "detection": valid DetectionResult
        """
        if len(samples) < 3:
            raise ValueError("At least 3 valid samples are required for calibration.")

        calibration_inputs = []

        for i, s in enumerate(samples):
            robot_pose = s.get("pose_matrix")
            detection = s.get("detection")

            if detection is None or not detection.valid():
                raise ValueError(f"Sample {i} has invalid detection.")

            if not isinstance(robot_pose, np.ndarray) or robot_pose.shape != (4, 4):
                raise ValueError(f"Sample {i} pose must be a 4x4 numpy array.")

            robot_pose = robot_pose.astype(np.float64)

            z_pose = zivid.calibration.Pose(robot_pose)

            calibration_inputs.append(
                zivid.calibration.HandEyeInput(z_pose, detection)
            )

        print("🧮 Running hand-eye calibration...")

        if self.hand_to_eye:
            print("➡️ Mode: Eye-to-Hand")
            result = zivid.calibration.calibrate_eye_to_hand(calibration_inputs)
        else:
            print("➡️ Mode: Eye-in-Hand")
            result = zivid.calibration.calibrate_eye_in_hand(calibration_inputs)

        if not result.valid():
            raise RuntimeError("❌ Calibration failed: invalid result.")

        transform = np.array(result.transform())
        print("✅ Calibration successful.\nTransform:\n", transform)

        return transform


# -----------------------------------------------------------------------------
# Example usage (safe test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    zi = ZividInterface(hand_to_eye=True, use_capture_assistant=False)

    detection, frame = zi.capture_calibration_board(
        save_debug_frame="debug.zdf"
    )

    if detection is None:
        print("⚠️ Try adjusting lighting or switching to manual settings.")
    else:
        print("👍 Ready to collect calibration samples.")
