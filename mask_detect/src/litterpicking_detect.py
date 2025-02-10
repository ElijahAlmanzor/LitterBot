#!/usr/bin/env python3

# ========================== HEADERS ==========================

import os
import sys
import json
import time
import warnings
import random
import numpy as np
import cv2
import torch
import rospy
import pyrealsense2
import tf2_ros
import tf2_geometry_msgs
import matplotlib.pyplot as plt

from configparser import Interpolation
from sklearn.decomposition import PCA
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PointStamped
from cv_bridge import CvBridge
from std_msgs.msg import String

# ========================== CONFIGURATIONS ==========================

warnings.filterwarnings("ignore")  # Suppress deprecation warnings
setup_logger()

# ROS Topics
CAMERA_TOPICS = {
    "color": "/camera1/color/image_raw",
    "depth": "/camera1/depth/image_rect_raw",
    "info": "/camera1/color/camera_info",
}

PUBLISH_TOPICS = {
    "largest_object_pixel": "/largest_objects_pixel",
    "largest_object_class": "/largest_objects_class",
}

# List of litter classes
LITTER_CLASSES = [
    "Aluminium foil", "Battery", "Aluminium blister pack", "Carded blister pack",
    "Other plastic bottle", "Clear plastic bottle", "Glass bottle", "Plastic bottle cap",
    "Metal bottle cap", "Broken glass", "Food Can", "Aerosol", "Drink can", "Toilet tube",
    "Other carton", "Egg carton", "Drink carton", "Corrugated carton", "Meal carton",
    "Pizza box", "Paper cup", "Disposable plastic cup", "Foam cup", "Glass cup",
    "Other plastic cup", "Food waste", "Glass jar", "Plastic lid", "Metal lid",
    "Other plastic", "Magazine paper", "Tissues", "Wrapping paper", "Normal paper",
    "Paper bag", "Plastified paper bag", "Plastic film", "Six pack rings", "Garbage bag",
    "Other plastic wrapper", "Single-use carrier bag", "Polypropylene bag", "Crisp packet",
    "Spread tub", "Tupperware", "Disposable food container", "Foam food container",
    "Other plastic container", "Plastic gloves", "Plastic utensils", "Pop tab",
    "Rope & strings", "Scrap metal", "Shoe", "Squeezable tube", "Plastic straw",
    "Paper straw", "Styrofoam piece", "Unlabeled litter", "Cigarette"
]

# ========================== CLASS DEFINITIONS ==========================

class LitterDetector:
    def __init__(self):
        """Initialize the Litter Detector."""
        self.camera_model = None
        self.color_image = None
        self.depth_image = None
        self.intrinsics = None
        self.transform = None
        self.bridge = CvBridge()
        self.predictor = None
        self.cfg = None
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # ROS Subscribers
        self.camera_info_subscriber = rospy.Subscriber(CAMERA_TOPICS["info"], CameraInfo, self.camera_info_callback)
        self.camera_depth_subscriber = rospy.Subscriber(CAMERA_TOPICS["depth"], Image, self.camera_depth_callback)
        self.camera_image_subscriber = rospy.Subscriber(CAMERA_TOPICS["color"], Image, self.camera_color_callback)

        # ROS Publishers
        self.object_pixel_publisher = rospy.Publisher(PUBLISH_TOPICS["largest_object_pixel"], Pose, queue_size=10)
        self.object_class_publisher = rospy.Publisher(PUBLISH_TOPICS["largest_object_class"], String, queue_size=10)

        # Load intrinsics & setup model
        self.get_intrinsics()
        self.setup_model()

    def get_intrinsics(self):
        """Retrieve camera intrinsics from ROS topic."""
        print("Waiting for camera intrinsics...")
        rospy.sleep(3)

        self.intrinsics = pyrealsense2.intrinsics()
        self.intrinsics.width = self.camera_model.width
        self.intrinsics.height = self.camera_model.height
        self.intrinsics.ppx = self.camera_model.K[2]
        self.intrinsics.ppy = self.camera_model.K[5]
        self.intrinsics.fx = self.camera_model.K[0]
        self.intrinsics.fy = self.camera_model.K[4]
        self.intrinsics.model = pyrealsense2.distortion.none
        self.intrinsics.coeffs = self.camera_model.D

    def camera_info_callback(self, data):
        """Update camera model data."""
        self.camera_model = data

    def camera_color_callback(self, data):
        """Update the latest color image from the camera."""
        self.color_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def camera_depth_callback(self, data):
        """Update the latest depth image from the camera."""
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def setup_model(self):
        """Set up the Detectron2 model for object detection."""
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        self.cfg.MODEL.WEIGHTS = os.path.expanduser("~/litterbot_ws/src/mask_detect/src/output/model_0004499.pth")
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LITTER_CLASSES)

        self.predictor = DefaultPredictor(self.cfg)

    def detect_largest_object(self):
        """Detect the largest object in the frame and publish its location & class."""
        outputs = self.predictor(self.color_image[..., ::-1])

        # Find the largest object
        mask_instances = np.asarray(outputs["instances"].pred_masks.to("cpu").numpy())
        largest_idx = np.argmax([np.count_nonzero(mask) for mask in mask_instances])

        # Get bounding box & class
        x_l, y_l, x_r, y_r = outputs["instances"].pred_boxes.tensor[largest_idx].cpu().numpy()
        class_id = outputs["instances"].pred_classes[largest_idx].cpu().numpy()
        class_name = LITTER_CLASSES[class_id]

        # Calculate center of bounding box
        x_centre = int((x_l + x_r) / 2)
        y_centre = int((y_l + y_r) / 2)

        # Get depth information
        mask_whole = mask_instances[largest_idx]
        depth_values = self.depth_image * mask_whole
        depth = np.sum(depth_values) / np.count_nonzero(depth_values) * 0.001  # Convert to meters

        # Publish object information
        largest_object_pose = Pose()
        largest_object_pose.position.x = x_centre
        largest_object_pose.position.y = y_centre
        self.object_pixel_publisher.publish(largest_object_pose)
        self.object_class_publisher.publish(class_name)

        print(f"Largest object: {class_name} at pixel ({x_centre}, {y_centre})")

    def main(self):
        """Continuously run object detection."""
        while not rospy.is_shutdown():
            if self.color_image is not None and self.depth_image is not None:
                self.detect_largest_object()
            rospy.sleep(0.1)


if __name__ == "__main__":
    rospy.init_node("litter_detector", anonymous=True)
    detector = LitterDetector()
    detector.main()
