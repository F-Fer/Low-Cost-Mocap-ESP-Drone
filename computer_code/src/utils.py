import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting


def plot_3d_points(points_3d, Rs=None, ts=None):
    """
    Plots the estimated 3D points and optionally camera poses using matplotlib.

    Args:
        points_3d: numpy array of 3D points (num_points, 3).
        Rs: Optional list/array of camera rotation matrices (num_cameras, 3, 3).
        ts: Optional list/array of camera translation vectors (num_cameras, 3 or 3x1).
            Assumes poses are relative to the world origin (Camera 0).
    """
    # Basic validation
    poses_provided = (Rs is not None and ts is not None)
    if poses_provided and len(Rs) != len(ts):
        raise ValueError("Number of rotation matrices and translation vectors must match.")

    if points_3d.shape[0] == 0 and not poses_provided:
        print("No valid points or poses to plot.")
        return

    fig = plt.figure(figsize=(10, 10)) # Make figure larger
    ax = fig.add_subplot(111, projection='3d')

    min_coord = np.array([np.inf, np.inf, np.inf])
    max_coord = np.array([-np.inf, -np.inf, -np.inf])

    # Plot 3D points if available
    if points_3d.shape[0] > 0:
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        ax.scatter(x, y, z, c='r', marker='o', s=20, label='3D Points') # Smaller points
        min_coord = np.minimum(min_coord, points_3d.min(axis=0))
        max_coord = np.maximum(max_coord, points_3d.max(axis=0))
    else:
        # Default coordinates if only plotting poses
        x, y, z = np.array([]), np.array([]), np.array([])
        min_coord = np.array([-1, -1, -1]) # Initialize reasonable bounds
        max_coord = np.array([1, 1, 1])


    # Plot Camera Poses if available
    cam_centers = []
    if poses_provided:
        axis_length = 0.5 # Length of the coordinate axes for cameras
        num_cameras = len(Rs)
        for i in range(num_cameras):
            R = Rs[i]
            t = ts[i].reshape(3, 1) # Ensure t is 3x1

            # Calculate camera center in world coordinates C = -R^T * t
            cam_center = -R.T @ t
            cam_center_flat = cam_center.flatten()
            cam_centers.append(cam_center_flat)

            # Get axes directions (columns of R.T)
            x_axis = R.T[:, 0]
            y_axis = R.T[:, 1]
            z_axis = R.T[:, 2] # Viewing direction

            # Draw axes using cam_center_flat
            ax.quiver(cam_center_flat[0], cam_center_flat[1], cam_center_flat[2],
                      x_axis[0], x_axis[1], x_axis[2], length=axis_length, color='red', label='X' if i == 0 else "")
            ax.quiver(cam_center_flat[0], cam_center_flat[1], cam_center_flat[2],
                      y_axis[0], y_axis[1], y_axis[2], length=axis_length, color='green', label='Y' if i == 0 else "")
            ax.quiver(cam_center_flat[0], cam_center_flat[1], cam_center_flat[2],
                      z_axis[0], z_axis[1], z_axis[2], length=axis_length, color='blue', label='Z (View)' if i == 0 else "")
            ax.text(cam_center_flat[0], cam_center_flat[1], cam_center_flat[2], f' C{i}', size=10, zorder=1, color='k')

        cam_centers = np.array(cam_centers)
        # Update min/max coordinates to include camera centers
        if cam_centers.shape[0] > 0:
            min_coord = np.minimum(min_coord, cam_centers.min(axis=0))
            max_coord = np.maximum(max_coord, cam_centers.max(axis=0))


    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Estimated 3D Points and Camera Poses')

    # --- Set Axis Limits ---
    # Calculate the range needed to encompass points and cameras
    center = (min_coord + max_coord) / 2.0
    # Handle cases where min/max might be inf if points_3d was empty initially
    if not np.all(np.isfinite(min_coord)) or not np.all(np.isfinite(max_coord)):
         print("Warning: Non-finite limits detected, defaulting plot range.")
         center = np.zeros(3)
         max_range = 2.0
    else:
         max_range = (max_coord - min_coord).max()

    if max_range == 0: max_range = 2.0 # Avoid zero range if single point/cam

    # Ensure equal aspect ratio
    half_range = max_range * 0.6 # Add some padding
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    # Add world origin axes for reference if camera 0 wasn't explicitly plotted at origin
    plot_origin = True
    if poses_provided and len(Rs) > 0:
         # Check if cam 0 pose is identity/zero (within tolerance)
         R0_is_identity = np.allclose(Rs[0], np.eye(3))
         t0_is_zero = np.allclose(ts[0], np.zeros((3,1))) # Check against 3x1 zero vector
         if R0_is_identity and t0_is_zero:
              plot_origin = False # Cam 0 already represents world origin

    if plot_origin:
        origin_axis_len = half_range * 0.1
        ax.quiver(0, 0, 0, 1, 0, 0, length=origin_axis_len, color='grey', arrow_length_ratio=0.3, label='World X')
        ax.quiver(0, 0, 0, 0, 1, 0, length=origin_axis_len, color='grey', arrow_length_ratio=0.3, label='World Y')
        ax.quiver(0, 0, 0, 0, 0, 1, length=origin_axis_len, color='grey', arrow_length_ratio=0.3, label='World Z')

    ax.legend()
    print("Displaying 3D plot...")
    plt.show()