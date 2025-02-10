#!/usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from math import pi
from kg_robot import kg_robot as kgr
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PoseArray
from geometry_msgs.msg import Pose

# ROS Topics
BBOX_TO_WORLD_COORDINATES = "/bbox_pixel_to_world_coordinates"
CAMERA_TOPIC_NAME = "/camera1/color/image_raw"

# Global Variables
bbox_location = None
camera_image = None
bridge = CvBridge()  # OpenCV Bridge

# ========================== CALLBACKS ==========================

def pose_callback(data):
    """Updates the bounding box location received from ROS."""
    global bbox_location
    bbox_location = data

def image_callback(data):
    """Updates the latest camera image received from ROS."""
    global camera_image
    camera_image = data

# ========================== MAIN FUNCTION ==========================

def main():
    global bbox_location, camera_image, bridge

    # Initialize ROS Node
    rospy.init_node("controller", anonymous=True)

    print("------------ Configuring Planter Planter (PP) -------------\n")
    burt = kgr.kg_robot(port=30010, db_host="192.168.2.10")

    # ROS Subscribers
    rospy.Subscriber(BBOX_TO_WORLD_COORDINATES, PoseArray, pose_callback)
    rospy.Subscriber(CAMERA_TOPIC_NAME, Image, image_callback)

    # Move to Home Position
    home_pose = Pose()
    home_pose.position.x, home_pose.position.y, home_pose.position.z = -0.010, -0.75, 0.75
    home_pose.orientation.x, home_pose.orientation.y = 0.0132, -0.0318
    home_pose.orientation.z, home_pose.orientation.w = -0.9994, 0.0078

    home_pose_robot = burt.ros_to_robot_pose_transform_no_rotation(home_pose)
    burt.movel(home_pose_robot)

    print("Moving to the home position...")
    
    # ========================== GRID GENERATION ==========================

    # Define Start & End Coordinates
    x_start, y_start, z_start = -0.14, -0.95, 0.5
    x_end, y_end, z_end = 0.120, -0.550, 0.8

    # Generate grid points using linspace
    x_coords = np.linspace(x_start, x_end, num=3)
    y_coords = np.linspace(y_start, y_end, num=3)
    z_coords = np.linspace(z_start, z_end, num=3)

    # Generate coordinate grid
    coordinates = [(x, y, z) for z in z_coords for x in x_coords for y in y_coords]

    # ========================== IMAGE CAPTURE & ROBOT MOVEMENT ==========================

    image_count = 502  # Starting index for image numbering
    filename_prefix = "image_"

    for x, y, z in coordinates:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.x, pose.orientation.y = 0.0132, -0.0318
        pose.orientation.z, pose.orientation.w = -0.9994, 0.0078

        # Wait until a valid camera image is received
        if camera_image is None:
            rospy.loginfo(f"ERROR: No camera image received yet on {CAMERA_TOPIC_NAME}")
        else:
            try:
                print(f"Saving Image {image_count}...")
                cv2_img = bridge.imgmsg_to_cv2(camera_image, "bgr8")
                filename = f"{filename_prefix}{image_count}.jpg"
                cv2.imwrite(filename, cv2_img)
                image_count += 1
            except CvBridgeError:
                print("Error converting image.")

        # Move Robot
        pose_robot = burt.ros_to_robot_pose_transform_no_rotation(pose)
        burt.movel(pose_robot, acc=2, vel=2)
        print(f"Moving to: {pose}")
        rospy.sleep(0.2)

    # Shutdown Procedure
    print("Task Complete. Closing connection...")
    burt.close()

# ========================== SCRIPT ENTRY POINT ==========================

if __name__ == "__main__":
    main()
