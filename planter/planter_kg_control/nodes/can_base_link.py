#!/usr/bin/env python3

import rospy
import numpy as np
import math
import tf
import tf2_ros

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Point
from scipy.spatial.transform import Rotation as R
import rtde_receive

# ========================== CONFIGURATIONS ==========================

TCP_WORLD_COORDINATES = "/tcp_world_coordinates"
UR_IP = "192.168.2.10"  # Update with your UR robot's IP address
CAMERA_OFFSETS = {
    "H": 0.397,  # Length of the bar (m)
    "h": -0.034,  # Offset of camera position (m)
    "w": 0.038,  # Offset of camera (m)
    "dy": 0.035,  # Y position offset (m)
}

# ========================== HELPER FUNCTIONS ==========================

def euler_to_quaternion(euler):
    """Convert Euler angles to Quaternion."""
    return R.from_euler("xyz", euler).as_quat()

def quaternion_to_euler(quat):
    """Convert Quaternion to Euler angles."""
    return R.from_quat(quat).as_euler("xyz")

# ========================== MAIN FUNCTION ==========================

def main():
    """Continuously broadcast the TCP pose of the UR robot."""
    rospy.init_node("ee_base_link", anonymous=True)

    # Transformation matrices
    RTrv = R.from_euler("y", -90, degrees=True) * R.from_euler("z", -90, degrees=True)
    rvTR = RTrv.inv()

    # Initialize robot connection & ROS publisher
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
    tcp_publisher = rospy.Publisher(TCP_WORLD_COORDINATES, Point, queue_size=10)
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    while not rospy.is_shutdown():
        pose = rtde_r.getActualTCPPose()

        # Convert UR pose to ROS coordinates
        ROS_pose_x, ROS_pose_y, ROS_pose_z = -pose[1], pose[0], pose[2]

        # Convert rotation vector to Euler angles (UR format)
        robot_ori = R.from_rotvec(pose[3:]).as_euler("xyz")

        # Convert to RViz right-handed coordinates
        ori_rviz_rh = rvTR * R.from_euler("xyz", robot_ori)
        ori_rviz = ori_rviz_rh.as_euler("xyz", degrees=False)

        # Convert to quaternion
        quat = euler_to_quaternion([ori_rviz[1], ori_rviz[0], ori_rviz[2]])

        # Compute camera offset transformation
        angle = np.pi / 2 - ori_rviz[2]
        dx = np.cos(angle) * (CAMERA_OFFSETS["H"] + CAMERA_OFFSETS["h"]) + np.sin(angle) * CAMERA_OFFSETS["w"]
        dz = np.sin(angle) * (CAMERA_OFFSETS["H"] + CAMERA_OFFSETS["h"]) - np.cos(angle) * CAMERA_OFFSETS["w"]

        # Publish TCP pose in world coordinates
        point = Point(ROS_pose_x, ROS_pose_y, ROS_pose_z)
        tcp_publisher.publish(point)

        # Broadcast transformation
        static_transform = TransformStamped()
        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "base_link"
        static_transform.child_frame_id = "overhead_camera"
        static_transform.transform.translation.x = ROS_pose_x + dx
        static_transform.transform.translation.y = ROS_pose_y + CAMERA_OFFSETS["dy"]
        static_transform.transform.translation.z = ROS_pose_z + dz
        static_transform.transform.rotation.x, static_transform.transform.rotation.y, \
            static_transform.transform.rotation.z, static_transform.transform.rotation.w = quat

        broadcaster.sendTransform(static_transform)
        print(f"Broadcasting transformation:\n{static_transform}")

if __name__ == "__main__":
    main()
