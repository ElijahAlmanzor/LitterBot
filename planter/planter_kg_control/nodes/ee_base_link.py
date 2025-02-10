#!/usr/bin/env python3

import rospy
import numpy as np
import os
import time
import tf
import tf2_ros

from sensor_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import rtde_receive

# ========================== CONFIGURATIONS ==========================

TCP_WORLD_COORDINATES = "/tcp_world_coordinates"
UR_IP = "192.168.2.10"  # Update with your UR robot's IP address
SAVE_PATH = os.path.join(os.path.expanduser("~"), "litter_ws", "src", "trajectories", "visual_servo.npy")

# Camera Pose (w.r.t. robot base)
CAMERA_POSE = [-0.2082, -0.5363, 0.2101, 0.0, 0.0, 0.0]  # X, Y, Z, RX, RY, RZ

# ========================== MAIN FUNCTION ==========================

def main():
    """Continuously track the UR robot TCP and publish its transformation."""
    rospy.init_node("ee_base_link", anonymous=True)

    # Initialize robot interface & ROS publisher
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
    tcp_publisher = rospy.Publisher(TCP_WORLD_COORDINATES, Point, queue_size=10)
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Rotation transformations
    RTrv = R.from_euler("y", -90, degrees=True) * R.from_euler("z", -90, degrees=True)
    rvTR = RTrv.inv()

    trajectory = []
    start_time = time.time()

    while not rospy.is_shutdown():
        # Get current TCP pose from robot
        current_pos = rtde_r.getActualTCPPose()
        point = Point(*current_pos[:3])
        tcp_publisher.publish(point)

        # Save trajectory point with timestamp
        trajectory.append([current_pos[0], current_pos[1], current_pos[2], time.time() - start_time])

        # Save trajectory data to file
        with open(SAVE_PATH, "wb") as f:
            np.save(f, trajectory)

        # Compute camera pose in ROS base frame
        ros_pose_x, ros_pose_y, ros_pose_z = -CAMERA_POSE[1], CAMERA_POSE[0], CAMERA_POSE[2]
        robot_ori = R.from_rotvec(CAMERA_POSE[3:]).as_euler("xyz")
        ori_rviz = rvTR * R.from_euler("xyz", robot_ori)
        quat = R.from_euler("xyz", [-ori_rviz.as_euler("xyz")[1], ori_rviz.as_euler("xyz")[0], ori_rviz.as_euler("xyz")[2]]).as_quat()

        # Broadcast transformation
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = "base_link"
        transform_msg.child_frame_id = "overhead_camera"
        transform_msg.transform.translation.x = ros_pose_x
        transform_msg.transform.translation.y = ros_pose_y
        transform_msg.transform.translation.z = ros_pose_z
        transform_msg.transform.rotation.x, transform_msg.transform.rotation.y, transform_msg.transform.rotation.z, transform_msg.transform.rotation.w = quat

        broadcaster.sendTransform(transform_msg)
        print(f"Broadcasting transformation: {point}")

        rospy.sleep(0.01)

if __name__ == "__main__":
    main()
