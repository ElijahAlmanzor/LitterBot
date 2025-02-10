#!/usr/bin/env python3

import time
import serial
import numpy as np
import os
import rospy
import cv2
import pyrealsense2
import tf2_ros
import tf2_geometry_msgs

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, TransformStamped
from std_msgs.msg import String

# RTDE Control & Receive
import rtde_receive
import rtde_control

# ========================== CONFIGURATIONS ==========================

# ROS Topics
LARGEST_OBJECT_PIXELS = "/largest_objects_pixel"
LARGEST_OBJECT_CLASS = "/largest_objects_class"
TCP_WORLD_COORDINATES = "/tcp_world_coordinates"

# UR Robot Connection
UR_IP = "192.168.2.10"

# Arduino Connection (Gripper)
ARDUINO_PORT = "/dev/ttyACM0"
GRIPPER_OPEN = "0"
GRIPPER_CLOSE = "255"

# Robot Speeds
VEL = 1
ACC = 1

# Predefined Positions (X, Y, Z, RX, RY, RZ)
HOME_POSITION = [-0.033, -0.83, 0.2149, 2.326, -2.1882, 0.042]
RECYCLE_POSITION = [-0.369, -0.444, -0.2122, 2.326, -2.1882, 0.042]
WASTE_POSITION = [-0.676, -0.444, -0.2122, 2.326, -2.1882, 0.042]

# Recycling Keywords
RECYCLE_KEYWORDS = {
    "Aluminium foil", "Other plastic bottle", "Clear plastic bottle", "Glass bottle",
    "Plastic bottle cap", "Metal bottle cap", "Broken glass", "Food Can", "Aerosol",
    "Drink can", "Toilet tube", "Other carton", "Egg carton", "Drink carton", 
    "Corrugated carton", "Meal carton", "Pizza box", "Paper cup", "Glass cup",
    "Glass jar", "Plastic lid", "Metal lid", "Tissues", "Normal paper", "Paper bag",
    "Six pack rings", "Tupperware", "Pop tab", "Rope & strings", "Scrap metal"
}

# ========================== CLASS DEFINITION ==========================

class LitterOperation:
    def __init__(self):
        """Initialize the litter pick-and-place operation."""
        rospy.init_node("pick_operation", anonymous=True)

        # State Variables
        self.color_image = None
        self.largest_object_pixel = None
        self.largest_object_class = None
        self.tcp_location = None
        self.record_trajectory = False
        self.trajectory_points = []
        self.trajectory_start_time = None

        # Hardware Interfaces
        self.rtde_c = rtde_control.RTDEControlInterface(UR_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
        self.arduino = serial.Serial(ARDUINO_PORT, 9600)
        self.bridge = CvBridge()

        # ROS Subscribers
        rospy.Subscriber(LARGEST_OBJECT_PIXELS, Pose, self.pose_callback)
        rospy.Subscriber(LARGEST_OBJECT_CLASS, String, self.class_callback)
        rospy.Subscriber(TCP_WORLD_COORDINATES, Point, self.tcp_callback)

        # Allow connections to establish
        rospy.sleep(3)

    # ========================== CALLBACK FUNCTIONS ==========================

    def pose_callback(self, data):
        """Callback for object pixel coordinates."""
        self.largest_object_pixel = data

    def class_callback(self, data):
        """Callback for detected object class."""
        self.largest_object_class = data.data

    def tcp_callback(self, data):
        """Callback for tracking end-effector position."""
        self.tcp_location = data

        if self.record_trajectory:
            current_time = time.time() - self.trajectory_start_time
            self.trajectory_points.append([data.x, data.y, data.z, current_time])
            print("Trajectory Recorded:", self.trajectory_points[-1])

    # ========================== PICK & PLACE LOGIC ==========================

    def move_to(self, position):
        """Move robot to a specified position."""
        self.rtde_c.moveL(position, VEL, ACC)

    def operate_gripper(self, state):
        """Open or close the gripper via Arduino."""
        self.arduino.write(state.encode())
        rospy.sleep(2)

    def start_trajectory_tracking(self):
        """Begin tracking robot's trajectory."""
        self.trajectory_start_time = time.time()
        self.record_trajectory = True

    def visual_servoing(self):
        """Align the robot with the detected object using visual servoing."""
        x_target, y_target = 376, 263
        kp = 0.001  # Proportional gain
        dt = 1 / 500  # Servoing update rate

        while True:
            start_time = time.time()

            # Compute errors
            x_error = self.largest_object_pixel.position.x - x_target
            y_error = self.largest_object_pixel.position.y - y_target

            # Compute velocity commands
            speed = [-kp * x_error, kp * y_error, 0, 0, 0, 0]
            self.rtde_c.speedL(speed, 0.1, dt)

            # Exit condition
            if abs(x_error) < 2 and abs(y_error) < 2:
                print("Visual Servoing Completed.")
                break

            # Ensure loop runs at correct speed
            duration = time.time() - start_time
            if duration < dt:
                rospy.sleep(dt - duration)

        # Stop movement
        self.rtde_c.speedL([0, 0, 0, 0, 0, 0], 0.1, dt)
        self.rtde_c.speedStop()

    def pick_place(self):
        """Perform the complete pick-and-place operation."""
        print("Moving to Home Position")
        self.move_to(HOME_POSITION)
        self.operate_gripper(GRIPPER_OPEN)

        # Perform visual servoing to align
        print("Starting Visual Servoing...")
        self.visual_servoing()

        # Move down to pick the object
        print("Moving to Object Location")
        object_pose = self.rtde_r.getActualTCPPose()
        object_pose[2] = -0.696  # Adjust Z position for grasping
        self.move_to(object_pose)

        # Close gripper to grasp the object
        self.operate_gripper(GRIPPER_CLOSE)

        # Lift the object
        object_pose[2] = -0.212
        self.move_to(object_pose)

        # Determine if object is recyclable
        destination = RECYCLE_POSITION if self.largest_object_class in RECYCLE_KEYWORDS else WASTE_POSITION

        # Move to disposal bin
        print(f"Moving to {'Recycling' if self.largest_object_class in RECYCLE_KEYWORDS else 'Waste'} Bin")
        self.move_to(destination)

        # Release object
        self.operate_gripper(GRIPPER_OPEN)

        # Return to home position
        self.move_to(HOME_POSITION)

    # ========================== MAIN LOOP ==========================

    def main(self):
        """Main loop for object detection and pick-and-place execution."""
        self.start_trajectory_tracking()
        objects_left = True

        while objects_left:
            self.largest_object_pixel = None
            self.largest_object_class = None

            print("Waiting for Object Detection...")
            timer = time.time()
            while self.largest_object_pixel is None or self.largest_object_class is None:
                rospy.sleep(1)
                if time.time() - timer > 5:
                    print("No more detected objects.")
                    objects_left = False
                    break

            if objects_left:
                self.pick_place()

        print(f"Total Execution Time: {time.time() - self.trajectory_start_time:.2f} seconds")

# ========================== EXECUTION ==========================

if __name__ == "__main__":
    LitterOperation().main()
