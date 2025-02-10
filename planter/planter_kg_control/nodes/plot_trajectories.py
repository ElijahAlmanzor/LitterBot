#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================== CONFIGURATIONS ==========================

# Path to Trajectories
TRAJ_PATH = os.path.join(os.path.expanduser("~"), "litter_ws", "src", "trajexp12")

# Number of trajectories to load
NUM_TRAJECTORIES = 5

# ========================== MAIN FUNCTION ==========================

def load_trajectories():
    """Loads trajectory data from .npy files and returns a list of NumPy arrays."""
    trajectories = []
    for num in range(NUM_TRAJECTORIES):
        file_path = os.path.join(TRAJ_PATH, f"{num}.npy")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                trajectories.append(np.load(f))
        else:
            print(f"Warning: File {file_path} does not exist.")
    return trajectories

def plot_trajectories(trajectories):
    """Plots Cartesian coordinates (X, Y, Z) against time."""
    
    # Check if any trajectories were loaded
    if not trajectories:
        print("No trajectory data available to plot.")
        return
    
    # Define Colors
    colors = ["k", "r", "b", "c", "y"]
    labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]

    # ==================== Figure 1: X, Y, Z vs Time ====================

    fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
    fig.suptitle("TCP Cartesian Coordinates (X, Y, Z) vs Time")

    for i, traj in enumerate(trajectories):
        ax[0].plot(traj[:, 3], traj[:, 0], f"{colors[i]}--", label=labels[i])
        ax[1].plot(traj[:, 3], traj[:, 1], f"{colors[i]}--")
        ax[2].plot(traj[:, 3], traj[:, 2] - 0.4, f"{colors[i]}--")

    ax[0].set_ylabel("X (m)")
    ax[1].set_ylabel("Y (m)")
    ax[2].set_ylabel("Z (m)")
    ax[2].set_xlabel("Time (s)")
    ax[0].legend(loc="upper right")

    # ==================== Figure 2: 3D Plot ====================

    fig2 = plt.figure(figsize=(8, 6))
    fig2.suptitle("TCP Cartesian Coordinates in 3D")
    ax2 = fig2.add_subplot(111, projection="3d")

    for i, traj in enumerate(trajectories):
        ax2.plot3D(traj[:, 0], traj[:, 1], traj[:, 2] - 0.4, f"{colors[i]}--", label=labels[i])

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.legend()

    # ==================== Show Plots ====================
    plt.show()

def main():
    trajectories = load_trajectories()
    plot_trajectories(trajectories)

# ========================== EXECUTION ==========================

if __name__ == "__main__":
    main()
