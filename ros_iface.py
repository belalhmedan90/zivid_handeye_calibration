import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import tf2_ros
from geometry_msgs.msg import Transform


def transform_to_matrix(transform: Transform) -> np.ndarray:
    trans = np.array(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
    )
    rot = Rotation.from_quat(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    ).as_matrix()
    matrix = np.identity(4)
    matrix[:3, 3] = trans
    matrix[:3, :3] = rot
    return matrix

def get_tf_mat(
    src_frame: str, tgt_frame: str
) -> np.ndarray:

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    try:
        tf_ = tfBuffer.lookup_transform(
            src_frame,
            tgt_frame,
            rospy.Time(0),
            rospy.Duration(2.0),
        )
    except (
        rospy.ROSException,
        rospy.ROSInterruptException,
    ) as e:
        rospy.logerr(str(e))
        raise SystemExit()
    tf_mat = transform_to_matrix(tf_.transform)
    return tf_mat
