#!/usr/bin/env python3

# ========================== HEADERS ==========================

import os
import sys
import json
import time
import random
import warnings
import numpy as np
import cv2
import rospy
import torch
import torchvision
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ========================== CONFIGURATIONS ==========================

warnings.filterwarnings("ignore")  # Suppress deprecation warnings

setup_logger()

# ROS Topics
OVERHEAD_CAMERA_TOPIC_NAME = "/camera1/color/image_raw"

# Global variables
image = None
bridge = None


# ========================== FUNCTION DEFINITIONS ==========================

def camera_image_callback(data):
    """Callback function to update the global image variable from ROS."""
    global image
    image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")


def get_litter_dicts(img_dir):
    """Loads dataset annotations and formats them for Detectron2."""
    json_file = os.path.join(img_dir, "taco_dataset.json")
    
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {
            "file_name": os.path.join(img_dir, v["filename"]),
            "image_id": idx,
            "height": cv2.imread(os.path.join(img_dir, v["filename"])).shape[0],
            "width": cv2.imread(os.path.join(img_dir, v["filename"])).shape[1],
            "annotations": [
                {
                    "bbox": [np.min(anno["shape_attributes"]["all_points_x"]),
                             np.min(anno["shape_attributes"]["all_points_y"]),
                             np.max(anno["shape_attributes"]["all_points_x"]),
                             np.max(anno["shape_attributes"]["all_points_y"])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [list(sum(zip(anno["shape_attributes"]["all_points_x"],
                                                  anno["shape_attributes"]["all_points_y"]), ()))],
                    "category_id": 0,
                }
                for anno in v["regions"]
            ],
        }
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    """Main function to run litter detection."""
    global image, bridge
    rospy.init_node("detectron_stream_test", anonymous=True)

    bridge = CvBridge()
    rospy.Subscriber(OVERHEAD_CAMERA_TOPIC_NAME, Image, camera_image_callback)
    rospy.sleep(3)  # Allow time for the first image to be received

    # Register dataset and metadata
    for d in ["train", "val"]:
        DatasetCatalog.register(f"litter_{d}", lambda d=d: get_litter_dicts(f"litte/{d}"))
        MetadataCatalog.get(f"litter_{d}").set(thing_classes=["Litter"])

    litter_metadata = MetadataCatalog.get("litter_train")

    # Load model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.98
    cfg.MODEL.WEIGHTS = os.path.join(os.path.expanduser("~"), "pp_ws", "src", "mask_detect", "src", "output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single-class prediction

    predictor = DefaultPredictor(cfg)

    while not rospy.is_shutdown():
        if image is None:
            continue

        outputs = predictor(image[..., ::-1])

        # Visualize results
        visualizer = Visualizer(image[:, :, ::-1], metadata=litter_metadata, scale=1.0)
        out_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[..., ::-1]

        try:
            # Find the largest litter object
            mask_instances = np.asarray(outputs["instances"].pred_masks.to("cpu").numpy())
            largest_litter_index = np.argmax([np.count_nonzero(mask) for mask in mask_instances])

            # Get bounding box coordinates
            x_l, y_l, x_r, y_r = outputs["instances"].pred_boxes.tensor[largest_litter_index].cpu().numpy()
            segmented_image = image[int(y_l):int(y_r), int(x_l):int(x_r)]

            # Obtain binary mask
            mask = mask_instances[largest_litter_index][int(y_l):int(y_r), int(x_l):int(x_r)]
            disp_mask = (mask * 255).astype(np.uint8)

            # Perform PCA for object orientation
            ypc, xpc = (y_r - y_l) // 2, (x_r - x_l) // 2
            one_indices = np.column_stack(np.where(mask == 1))
            one_indices[:, 0] = ypc - one_indices[:, 0]
            one_indices[:, 1] = one_indices[:, 1] - xpc

            pca = PCA(n_components=1).fit(one_indices)
            angle = np.arctan2(pca.components_[0][1], pca.components_[0][0]) * 180 / np.pi

            print(f"Litter Orientation: {angle:.2f}°")

            # Display results
            cv2.imshow("Segmented Mask", disp_mask)
            cv2.imshow("Detection", out_image)

        except Exception as e:
            print(f"Error processing litter detection: {e}")
            cv2.imshow("Detection", out_image)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
