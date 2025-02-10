#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Point
from scipy.spatial.transform import Rotation as R
import rtde_receive

# ========================== CONFIGURATIONS ==========================

# ROS Topics
TCP_WORLD_COORDINATES = "/tcp_world_coordinates"

# UR Robot Connection
UR_IP = "192.168.2.10"  # Change if necessary

# ========================== MAIN FUNCTION ==========================

def main():
    """Broadcasts the transformation between the UR robot's base and camera using RTDE."""

    rospy.init_node("ee_base_link", anonymous=True)

    # RTDE Connection
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
    
    # ROS Publisher & TF Broadcaster
    tcp_publisher = rospy.Publisher(TCP_WORLD_COORDINATES, Point, queue_size=10)
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Transformation from Camera to ROS Base (Euler Angles to Rotation)
    RvTcam = R.from_euler("xyz", [180, -180, -180], degrees=True)
    
    while not rospy.is_shutdown():
        # Read TCP Pose from UR Robot
        tcp_pose = rtde_r.getActualTCPPose()  # Format: [X, Y, Z, RX, RY, RZ]

        # Prepare ROS Transform Message
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = "base_link"
        transform_msg.child_frame_id = "overhead_camera"

        # Assign Position
        transform_msg.transform.translation.x = tcp_pose[0]
        transform_msg.transform.translation.y = tcp_pose[1]
        transform_msg.transform.translation.z = tcp_pose[2]

        # Convert Orientation (TCP Rotation Vector to Quaternion)
        robot_ori = R.from_rotvec(tcp_pose[3:])  # Convert from rotation vector
        ori_rviz = RvTcam * robot_ori  # Convert to ROS coordinate frame
        quat = ori_rviz.as_quat()  # Convert to quaternion

        transform_msg.transform.rotation.x = quat[0]
        transform_msg.transform.rotation.y = quat[1]
        transform_msg.transform.rotation.z = quat[2]
        transform_msg.transform.rotation.w = quat[3]

        # Broadcast Transformation
        broadcaster.sendTransform(transform_msg)

        # Publish TCP Position
        tcp_point = Point(x=tcp_pose[0], y=tcp_pose[1], z=tcp_pose[2])
        tcp_publisher.publish(tcp_point)

        rospy.loginfo(f"Broadcasting Transformation:\n{transform_msg}")
        rospy.sleep(0.1)  # Adjust rate if needed

# ========================== EXECUTION ==========================

if __name__ == "__main__":
    main()
