import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import Transform


# Global instances
_node: Node = None
_tf_buffer = None
_tf_listener = None
_spin_thread = None


def _spin_node():
    """Continuously spin the node in a background thread."""
    global _node
    rclpy.spin(_node)


def init_tf_handler():
    """Initialize TF buffer, listener, and start spinning."""
    global _node, _tf_buffer, _tf_listener, _spin_thread

    if _node is not None:
        return

    if not rclpy.ok():
        rclpy.init()

    _node = rclpy.create_node("tf_lookup_client")

    _tf_buffer = tf2_ros.Buffer()
    _tf_listener = tf2_ros.TransformListener(_tf_buffer, _node)

    # 🔥 CRITICAL FIX: spin in background
    _spin_thread = threading.Thread(target=_spin_node, daemon=True)
    _spin_thread.start()


def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Convert geometry_msgs Transform to 4x4 matrix."""
    trans = np.array([
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ])

    rot = Rotation.from_quat([
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans

    return T


def get_tf_mat(
    target_frame: str,
    source_frame: str,
    timeout_sec: float = 2.0,
) -> np.ndarray:
    """
    Get transformation matrix from source_frame -> target_frame.

    Returns:
        4x4 numpy array
    """
    global _node, _tf_buffer

    if _node is None:
        init_tf_handler()

    try:
        now = Time()

        # 🔥 Correct order in ROS2:
        # target_frame, source_frame
        tf_msg = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            now,
            timeout=Duration(seconds=timeout_sec),
        )

        return transform_to_matrix(tf_msg.transform)

    except Exception as e:
        if _node:
            _node.get_logger().error(f"TF lookup failed: {str(e)}")
        raise


def shutdown_tf_handler():
    """Clean shutdown (optional but recommended)."""
    global _node

    if _node is not None:
        _node.destroy_node()
        rclpy.shutdown()