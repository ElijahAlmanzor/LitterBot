#!/usr/bin/env python3

# ========================== IMPORTS ==========================

import os
import json
import time
import warnings
import numpy as np
import cv2
import torch
import rospy
import pyrealsense2
import tf2_ros
import tf2_geometry_msgs
import matplotlib.pyplot as plt

from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PointStamped
from cv_bridge import CvBridge
from sklearn.decomposition import PCA

# ========================== CONFIGURATIONS ==========================

warnings.filterwarnings("ignore")  # Suppress deprecation warnings
setup_logger()

# ROS Topics
CAMERA_TOPICS = {
    "color": "/camera1/color/image_raw",
    "depth": "/camera1/depth/image_rect_raw",
    "info": "/camera1/color/camera_info",
}

BBOX_TOPIC = "/bbox_pixel_to_world_coordinates"

# Model Parameters
MODEL_PATH = os.path.expanduser("~/pp_ws/src/mask_detect/src/output/model_final_bar.pth")
CONFIDENCE_THRESHOLD = 0.85
NUM_CLASSES = 2  # ["Jersey Royal", "Handle Bar"]

# ========================== CLASS DEFINITIONS ==========================

class LitterLocaliser:
    def __init__(self):
        """Initialize the Litter Localiser."""
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
        rospy.Subscriber(CAMERA_TOPICS["info"], CameraInfo, self.camera_info_callback)
        rospy.Subscriber(CAMERA_TOPICS["depth"], Image, self.camera_depth_callback)
        rospy.Subscriber(CAMERA_TOPICS["color"], Image, self.camera_color_callback)

        # ROS Publisher
        self.coordinates_publisher = rospy.Publisher(BBOX_TOPIC, Pose, queue_size=10)

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
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
        self.cfg.MODEL.WEIGHTS = MODEL_PATH
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

        self.predictor = DefaultPredictor(self.cfg)

    def detect_largest_litter(self):
        """Detect the largest piece of litter and publish its world coordinates."""
        outputs = self.predictor(self.color_image[..., ::-1])

        # Find the largest object
        mask_instances = np.asarray(outputs["instances"].pred_masks.to("cpu").numpy())
        largest_idx = np.argmax([np.count_nonzero(mask) for mask in mask_instances])

        # Get bounding box
        x_l, y_l, x_r, y_r = outputs["instances"].pred_boxes.tensor[largest_idx].cpu().numpy()

        # Calculate center of bounding box
        x_centre = int((x_l + x_r) / 2)
        y_centre = int((y_l + y_r) / 2)

        # Get depth information
        mask_whole = mask_instances[largest_idx]
        depth_values = self.depth_image * mask_whole
        depth = np.sum(depth_values) / np.count_nonzero(depth_values) * 0.001  # Convert to meters

        # Convert to world coordinates
        xyz_camera_frame = self.convert_depth_to_phys_coord(x_centre, y_centre, depth)
        xyz_world_frame = self.convert_camera_to_world_frame(xyz_camera_frame)

        # Publish litter position
        litter_pose = Pose()
        litter_pose.position.x = xyz_world_frame.point.x
        litter_pose.position.y = xyz_world_frame.point.y
        litter_pose.position.z = xyz_world_frame.point.z
        self.coordinates_publisher.publish(litter_pose)

        print(f"Largest litter detected at: ({xyz_world_frame.point.x:.3f}, {xyz_world_frame.point.y:.3f}, {xyz_world_frame.point.z:.3f})")

    def convert_camera_to_world_frame(self, point):
        """Transform point from camera frame to world frame."""
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = "overhead_camera"
        point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = point

        return tf2_geometry_msgs.do_transform_point(point_stamped, self.transform)

    def convert_depth_to_phys_coord(self, x, y, depth):
        """Convert depth image coordinates to physical world coordinates."""
        result = pyrealsense2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return result[2], -result[0], -result[1]  # Convert to ROS coordinates

    def main(self):
        """Continuously detect litter and publish its location."""
        while not rospy.is_shutdown():
            if self.color_image is not None and self.depth_image is not None:
                self.detect_largest_litter()
            rospy.sleep(0.1)


if __name__ == "__main__":
    rospy.init_node("litter_localiser", anonymous=True)
    litter_localiser = LitterLocaliser()
    litter_localiser.main()
