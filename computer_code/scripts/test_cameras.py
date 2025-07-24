from pseyepy import Camera
import cv2
import time
import numpy as np
from scipy import linalg
import json
from datetime import datetime
# BA Imports
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from utils import plot_3d_points

NUM_CAMERAS = 4
FPS = 60
THRESHOLD_VALUE = 40
EXPOSURE = 250
GAIN = 30

# Load camera parameters
f = open("/Users/finnferchau/dev/datfus/camera-params.json")
camera_params = json.load(f)


# Load camera parameters
KS = [np.array(camera_params[i]["intrinsic_matrix"]) for i in range(NUM_CAMERAS)]
DISTROTION_COEFFICIENTS = [np.array(camera_params[i]["distortion_coef"]) for i in range(NUM_CAMERAS)]
ROTATIONS = [np.array(camera_params[i]["rotation"]) for i in range(NUM_CAMERAS)]


def make_square(img):
    x, y, _ = img.shape
    size = max(x, y)
    new_img = np.zeros((size, size, 3), dtype=np.uint8)
    ax,ay = (size - img.shape[1])//2,(size - img.shape[0])//2
    new_img[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img

    # Pad the new_img array with edge pixel values
    # Apply feathering effect
    feather_pixels = 8
    for i in range(feather_pixels):
        alpha = (i + 1) / feather_pixels
        new_img[ay - i - 1, :] = img[0, :] * (1 - alpha)  # Top edge
        new_img[ay + img.shape[0] + i, :] = img[-1, :] * (1 - alpha)  # Bottom edge

    return new_img

def find_dots(img):
    """
    Find the brightest dots in the image.

    Args:
        img: numpy array of image. (H, W, 3)

    Returns:
        numpy array of image. (H, W, 3)
        numpy array of image_points. (num_points, 2)
    """

    # Convert to grayscale
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Threshold the image
    grey = cv2.threshold(grey, 255*0.2, 255, cv2.THRESH_BINARY)[1]
    # Find contours
    contours,_ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    img = cv2.drawContours(img, contours, -1, (0,255,0), 1)

    # Find dots and calculate their center
    image_points = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv2.putText(img, f'({center_x}, {center_y})', (center_x,center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
            cv2.circle(img, (center_x,center_y), 1, (100,255,100), -1)
            image_points.append([center_x, center_y])

    if len(image_points) == 0:
        image_points = [[None, None]]

    return img, image_points


def capture_points(cameras):
    """
    Capture the points from the cameras.

    Args:
        cameras: list of Camera objects.

    Returns:
        list of numpy arrays of image points. (num_points, num_cameras, 2)
    """

    kernel = np.array([[-2,-1,-1,-1,-2],
                        [-1,1,3,1,-1],
                        [-1,3,4,3,-1],
                        [-1,1,3,1,-1],
                        [-2,-1,-1,-1,-2]])

    list_of_points_per_camera = [] # (num_points, num_cameras, 2)
    print("Press Enter to capture points")
    while True:

        frames, _ = cameras.read()
        points_per_camera = [] # [num_cameras, 2]
        for i, frame in enumerate(frames):
            # Iterate over cameras
            frame = np.rot90(frame, k=ROTATIONS[i])
            frame = make_square(frame)
            frame = cv2.undistort(frame, KS[i], DISTROTION_COEFFICIENTS[i])
            frame = cv2.GaussianBlur(frame,(9,9),0)
            frame = cv2.filter2D(frame, -1, kernel)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Find dots
            frame, image_points = find_dots(frame)# [H, W, 3], [num_points, 2]
            image_points = image_points[0] # Only take the first point
            frames[i] = frame
            points_per_camera.append(image_points)

        # Display both frames side by side
        combined_frame = np.hstack(frames[:NUM_CAMERAS])
        cv2.imshow('PS3i Cameras - Detected Points', combined_frame)
        if cv2.waitKey(1) & 0xFF == 13:  # 13 is the keycode for Enter
            points_per_camera = np.array(points_per_camera)
            if len(list_of_points_per_camera) == 0:
                list_of_points_per_camera = points_per_camera.reshape(1, -1, 2)
            else:
                list_of_points_per_camera = np.append(list_of_points_per_camera, np.expand_dims(points_per_camera, axis=0), axis=0)
            print(f"Points: \n{list_of_points_per_camera.shape}")
            print("Press Enter to capture next points")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Points: \n{list_of_points_per_camera}")
            break
        time.sleep(1/FPS)
    
    cv2.destroyAllWindows()

    return list_of_points_per_camera


def estimate_relative_pose(image_points):
    """
    Estimate the relative pose using cv2.recoverPose and return inlier indices.

    Args:
        image_points: numpy array of image points. (num_points, num_cameras, 2)

    Returns:
        List of camera poses (dictionaries with 'R' and 't'), relative to camera 0.
        Indices of the inlier points used for the Camera 0 <-> Camera 1 pose estimation.
        Returns None, None if pose estimation fails for any required pair.
    """

    if image_points.shape[0] < 5 or image_points.shape[1] < NUM_CAMERAS:
        print(f"Error: Need at least 5 points and {NUM_CAMERAS} camera views. Got shape {image_points.shape}")
        return None, None

    # Initialize list to store poses, camera 0 is the reference
    camera_poses = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    inlier_indices_01 = None # Store inliers for cam0-cam1 pair

    points_cam0 = image_points[:, 0, :].astype(np.float32) # Points in reference camera
    K0 = KS[0] # Intrinsics for reference camera

    for i in range(1, NUM_CAMERAS):
        print(f"--- Estimating Pose for Camera {i} relative to Camera 0 ---")
        points_cami = image_points[:, i, :].astype(np.float32)
        original_indices = np.arange(image_points.shape[0])

        # --- Find Inliers between Camera 0 and Camera i ---
        F, mask_fundamental = cv2.findFundamentalMat(points_cam0, points_cami, method=cv2.FM_RANSAC, ransacReprojThreshold=1, confidence=0.99)

        if F is None:
            print(f"Error: Could not compute Fundamental Matrix between Cam 0 and Cam {i}.")
            return None, None # Essential failure

        fundamental_mask_bool = mask_fundamental.ravel() == 1
        inliers0 = points_cam0[fundamental_mask_bool]
        inliersi = points_cami[fundamental_mask_bool]
        indices_after_fundamental = original_indices[fundamental_mask_bool]

        if len(inliers0) < 5:
            print(f"Error: Not enough inliers ({len(inliers0)}) after F matrix calculation between Cam 0 and Cam {i}.")
            return None, None

        # Note: Essential matrix and recoverPose use K of the *first* camera in the pair (K0 here)
        E, mask_essential = cv2.findEssentialMat(inliers0, inliersi, K0, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            print(f"Error: Could not compute Essential Matrix between Cam 0 and Cam {i}.")
            return None, None

        # --- Use cv2.recoverPose to get relative pose Ri, ti of Cam i w.r.t Cam 0 ---
        points_in_front_count, Ri, ti, mask_pose = cv2.recoverPose(
            E, inliers0, inliersi, K0, mask=mask_essential
        )

        if Ri is None or ti is None:
            print(f"Error: cv2.recoverPose failed between Cam 0 and Cam {i}.")
            return None, None

        num_passed_chirality = np.sum(mask_pose) if mask_pose is not None else 0
        print(f"Cam 0 <-> Cam {i}: recoverPose finished. Points passing chirality: {num_passed_chirality} / {len(inliers0)}")

        if num_passed_chirality < 1: # Might need a higher threshold
            print(f"Warning: Very few points passed chirality check between Cam 0 and Cam {i}.")
            # Depending on strictness, could return None, None here

        # Store the calculated pose for camera i
        camera_poses.append({"R": Ri, "t": ti.reshape(3, 1)}) # Ensure t is column vector

        # If this was the first pair (i=1), store its inlier indices
        if i == 1:
            if mask_pose is not None:
                inlier_indices_01 = indices_after_fundamental[mask_pose.ravel() == 1]
            else:
                print("Warning: recoverPose did not return a mask for Cam 0-1.")
                inlier_indices_01 = indices_after_fundamental # Fallback?
            print(f"Stored {len(inlier_indices_01)} inlier indices from Cam 0-1 pair.")


    if inlier_indices_01 is None and NUM_CAMERAS > 1:
        print("Error: Failed to determine inliers for the Cam 0-1 pair.")
        return None, None

    print(f"Successfully estimated poses for all {NUM_CAMERAS} cameras.")
    return camera_poses, inlier_indices_01


# === Bundle Adjustment Functions ===

def project(points_3d, K, R, t, dist_coeffs):
    """Project 3D points into a camera view."""
    # cv2.projectPoints expects points_3d as (N, 1, 3) or (1, N, 3)
    # R should be Rodrigues vector or rotation matrix
    # t should be translation vector
    if R.shape == (3, 3): # Convert rotation matrix to Rodrigues vector
        rvec, _ = cv2.Rodrigues(R)
    else:
        rvec = R.reshape(3, 1) # Ensure it's a column vector if already Rodrigues

    tvec = t.reshape(3, 1)

    # Ensure points_3d is float64 or float32 for cv2.projectPoints
    points_3d_float = points_3d.astype(np.float64)

    points_2d_proj, _ = cv2.projectPoints(points_3d_float.reshape(-1, 1, 3), rvec, tvec, K, dist_coeffs)
    return points_2d_proj.reshape(-1, 2) # Reshape back to (N, 2)


def bundle_adjustment_residual(params, n_cameras, n_points, points_2d_observed, Ks, Dists):
    """
    Calculate reprojection errors for bundle adjustment (Multiple Cameras).

    Args:
        params: Flat array containing optimization parameters:
                - R_vec_cam1..N-1 (3 * (n_cameras-1)) : Rotation vectors for cameras 1 to N-1
                - t_vec_cam1..N-1 (3 * (n_cameras-1)) : Translation vectors for cameras 1 to N-1
                - points_3d_flat (3 * n_points) : Flattened 3D point coordinates
        n_cameras: Total number of cameras.
        n_points: Number of 3D points.
        points_2d_observed: Observed 2D points, shape (n_points, n_cameras, 2)
                           [point_idx, camera_idx, coord_xy]
                           Use NaN for missing observations if applicable.
        Ks: List of camera intrinsic matrices.
        Dists: List of distortion coefficients.

    Returns:
        Flattened array of residuals (reprojection errors).
    """
    # --- Unpack Parameters ---
    num_pose_params_per_cam = 6 # 3 for rvec, 3 for tvec
    num_static_params = 0 # No fixed params like intrinsics in this version
    # Calculate the index where pose parameters end
    pose_params_end_idx = num_static_params + num_pose_params_per_cam * (n_cameras - 1)
    pose_params = params[num_static_params : pose_params_end_idx]
    # The rest are the 3D points
    points_3d = params[pose_params_end_idx:].reshape((n_points, 3))

    # --- Define Camera Poses ---
    camera_poses = [{"R": np.eye(3), "t": np.zeros(3)}] # Cam 0 is reference
    for i in range(n_cameras - 1):
        # Extract rvec and tvec for camera i+1
        rvec = pose_params[i * num_pose_params_per_cam : i * num_pose_params_per_cam + 3]
        tvec = pose_params[i * num_pose_params_per_cam + 3 : (i + 1) * num_pose_params_per_cam]
        R, _ = cv2.Rodrigues(rvec) # Convert rotation vector to matrix
        camera_poses.append({"R": R, "t": tvec}) # Store pose

    # --- Calculate Residuals for each camera ---
    all_residuals = []
    for cam_idx in range(n_cameras):
        # Get parameters for the current camera
        K = Ks[cam_idx]
        dist = Dists[cam_idx] # Corrected variable name
        R = camera_poses[cam_idx]["R"]
        t = camera_poses[cam_idx]["t"]

        # Project the current estimate of 3D points into this camera
        points_2d_proj = project(points_3d, K, R, t, dist)
        # Get the corresponding observed 2D points
        observed = points_2d_observed[:, cam_idx, :] # Shape (N, 2)

        # Handle potential missing observations (if upstream filtering allows NaN)
        # Here we assume all observations passed to BA are valid based on main() filtering
        valid_obs_mask = np.all(np.isfinite(observed), axis=1)
        if not np.all(valid_obs_mask):
             # This case should ideally not happen if points_2d_observed_filtered is correct
             print(f"Warning: Invalid observed points detected for camera {cam_idx} within BA residual.")
             observed = observed[valid_obs_mask]
             points_2d_proj = points_2d_proj[valid_obs_mask]
             if observed.shape[0] == 0: continue # Skip camera if no valid points left

        # Calculate error: observed - projected
        errors = observed - points_2d_proj # Shape (N_valid, 2)
        all_residuals.append(errors) # Add errors for this camera

    # Flatten all valid residuals into a single 1D array
    # Order: [err_cam0_pt0_x, err_cam0_pt0_y, ..., err_camN-1_ptM_x, err_camN-1_ptM_y]
    flat_residuals = np.concatenate([res.flatten() for res in all_residuals])
    return flat_residuals

# === End Bundle Adjustment Functions ===


# === Multi-View Triangulation Functions ===

def triangulate_point_dlt(points_2d_all_views, camera_poses, Ks):
    """Triangulates a single 3D point from multiple 2D observations using DLT."""
    n_views = len(points_2d_all_views)
    if n_views != len(camera_poses) or n_views != len(Ks):
        raise ValueError("Number of points, poses, and intrinsics must match.")

    # Filter out invalid points (NaN/Inf) and corresponding poses/Ks
    valid_mask = np.all(np.isfinite(points_2d_all_views), axis=1)
    if np.sum(valid_mask) < 2:
        # print("Warning: Less than 2 valid views for triangulation.")
        return np.array([np.nan, np.nan, np.nan]) # Need at least 2 views

    points_2d_valid = points_2d_all_views[valid_mask]
    poses_valid = [camera_poses[i] for i, valid in enumerate(valid_mask) if valid]
    Ks_valid = [Ks[i] for i, valid in enumerate(valid_mask) if valid]

    # Construct projection matrices
    Ps = []
    for K, pose in zip(Ks_valid, poses_valid):
        R = pose["R"]
        t = pose["t"].reshape(3, 1)
        P = K @ np.hstack((R, t))
        Ps.append(P)

    # Construct the DLT matrix A
    A = []
    for P, point_2d in zip(Ps, points_2d_valid):
        x, y = point_2d[0], point_2d[1]
        A.append(y * P[2, :] - P[1, :])
        A.append(P[0, :] - x * P[2, :])
    A = np.array(A)

    # Solve for the 3D point using SVD on A
    # The solution is the null space of A, which corresponds to the singular vector
    # associated with the smallest singular value.
    U, s, Vh = linalg.svd(A)
    # The 3D point in homogeneous coordinates is the last row of Vh (or last col of V)
    point_4d_hom = Vh[-1]

    # Dehomogenize
    if abs(point_4d_hom[3]) < 1e-6: # Avoid division by zero
        # print("Warning: Division by zero during dehomogenization.")
        return np.array([np.nan, np.nan, np.nan])

    point_3d = point_4d_hom[:3] / point_4d_hom[3]
    return point_3d

def triangulate_points_dlt(points_2d_multi_cam, camera_poses, Ks):
    """Triangulates multiple 3D points from N camera views."""
    n_points = points_2d_multi_cam.shape[0]
    if n_points == 0:
        return np.empty((0, 3))

    points_3d = np.zeros((n_points, 3))
    for i in range(n_points):
        # points_2d_multi_cam[i] has shape (n_cameras, 2)
        points_3d[i] = triangulate_point_dlt(points_2d_multi_cam[i], camera_poses, Ks)

    return points_3d

# === End Multi-View Triangulation Functions ===


def main():
    # Initialize cameras if needed
    try:
        cameras = Camera(fps=FPS, resolution=Camera.RES_SMALL, gain=GAIN, exposure=EXPOSURE, colour=True)
        num_connected_cameras = len(cameras.read())
        print(f"Detected {num_connected_cameras} cameras.")
        # if num_connected_cameras < NUM_CAMERAS:
        #     print(f"Error: Need {NUM_CAMERAS} cameras, detected {num_connected_cameras}.\nUsing hardcoded points instead.")
        #     cameras = None
    except Exception as e:
        print(f"Error initializing cameras: {e}")
        print("Using hardcoded points instead.")
        cameras = None

    if cameras:
        list_of_points_per_camera = np.array(capture_points(cameras))
        print(f"list_of_points_per_camera: {list_of_points_per_camera}")
        # Save the captured points to a file
        np.save(f'captured_points_{time.strftime("%Y%m%d_%H%M%S")}.npy', list_of_points_per_camera)
        print("Saved captured points to captured_points.npy")
        cameras.end()
    else:
        # ---- Use Hardcoded points for testing (Ensure shape matches NUM_CAMERAS) ----
        # Example for NUM_CAMERAS = 3 (Replace with actual data if needed)
        if NUM_CAMERAS == 3:
             list_of_points_per_camera = np.array([[[256, 101], [184, 126], [100, 100]],
                                                   [[201, 170], [111, 196], [110, 180]],
                                                   [[248, 59],  [ 20, 91], [ 50, 70]],
                                                   [[218, 150], [223, 181], [200, 160]],
                                                   [[234, 120], [236, 146], [210, 130]],
                                                   [[282, 108], [173, 135], [150, 115]],
                                                   [[142, 121], [ 59, 144], [ 70, 130]],
                                                   [[124, 228], [ 56, 235], [ 80, 230]],
                                                   [[234, 217], [152, 255], [170, 240]],
                                                   [[244, 173], [ 94, 203], [120, 190]],
                                                   ], dtype=float)
        elif NUM_CAMERAS == 2:
             list_of_points_per_camera = np.array([[[256, 101], [184, 126]],
                                                [[201, 170], [111, 196]],
                                                [[248, 59],  [ 20, 91]],
                                                [[218, 150], [223, 181]],
                                                [[234, 120], [236, 146]],
                                                [[282, 108], [173, 135]],
                                                [[142, 121], [ 59, 144]],
                                                [[124, 228], [ 56, 235]],
                                                [[234, 217], [152, 255]],
                                                [[244, 173], [ 94, 203]],
                                                [[255, 167], [186, 203]],
                                                [[246, 143], [227, 175]],
                                                [[296, 112], [265, 139]],
                                                [[269, 134], [205, 166]],
                                                [[230, 149], [143, 177]],
                                                [[222, 204], [141, 237]],
                                                [[223, 214], [143, 248]],
                                                [[235, 176], [128, 207]],
                                                [[233, 126], [ 56, 153]],
                                                [[298, 124], [168, 156]]], dtype=float)
        elif NUM_CAMERAS == 4:
             list_of_points_per_camera = np.array([[[256, 101], [184, 126], [100, 100], [150, 120]],
                                                   [[201, 170], [111, 196], [110, 180], [160, 190]],
                                                   [[248, 59],  [ 20, 91], [ 50, 70], [100, 80]],
                                                   [[218, 150], [223, 181], [200, 160], [250, 170]],
                                                   [[234, 120], [236, 146], [210, 130], [260, 140]],
                                                   [[282, 108], [173, 135], [150, 115], [200, 125]],
                                                   [[142, 121], [ 59, 144], [ 70, 130], [120, 140]],
                                                   [[124, 228], [ 56, 235], [ 80, 230], [130, 240]],
                                                   [[234, 217], [152, 255], [170, 240], [220, 250]],
                                                   [[244, 173], [ 94, 203], [120, 190], [170, 200]],
                                                   [[255, 167], [186, 203], [160, 190], [210, 200]],
                                                   [[246, 143], [227, 175], [190, 160], [240, 170]],
                                                   [[296, 112], [265, 139], [220, 120], [270, 130]],
                                                   [[269, 134], [205, 166], [180, 150], [230, 160]],
                                                   [[230, 149], [143, 177], [150, 160], [200, 170]],
                                                   [[222, 204], [141, 237], [170, 220], [220, 230]],
                                                   [[223, 214], [143, 248], [180, 230], [230, 240]],
                                                   [[235, 176], [128, 207], [160, 190], [210, 200]],
                                                   [[233, 126], [ 56, 153], [100, 140], [150, 150]],
                                                   [[298, 124], [168, 156], [190, 140], [240, 150]]], dtype=float)
             list_of_points_per_camera = np.load("captured_points_20250416_112722.npy")

        else:
            print(f"Error: Hardcoded points only available for NUM_CAMERAS=2 - 4.")
            return
        # ---- End Hardcoded points ----

    if list_of_points_per_camera.shape[0] == 0:
        print("Exiting due to lack of points.")
        return
    if list_of_points_per_camera.shape[1] != NUM_CAMERAS:
         print(f"Error: Shape mismatch. Expected {NUM_CAMERAS} cameras, data has {list_of_points_per_camera.shape[1]}.")
         return


    initial_poses, inlier_indices_01 = estimate_relative_pose(list_of_points_per_camera)

    # Check if pose estimation was successful
    if initial_poses is None:
        print("Exiting because relative pose estimation failed.")
        return
    # Check if we got inliers for the crucial 0-1 pair
    if NUM_CAMERAS > 1 and inlier_indices_01 is None:
         print("Exiting because pose estimation failed to get inliers for Cam 0-1 pair.")
         return
    if NUM_CAMERAS > 1 and len(inlier_indices_01) == 0:
        print("Exiting because pose estimation returned 0 inliers for Cam 0-1 pair.")
        return

    # --- Initial Triangulation (using inliers from Cam 0-1 pair) ---
    if NUM_CAMERAS > 1:
        # Select only the inlier points (rows) across *all* cameras
        inlier_points_all_cams = list_of_points_per_camera[inlier_indices_01, :, :]
        n_inliers = inlier_points_all_cams.shape[0]
        print(f"Using {n_inliers} points (inliers from Cam 0-1) for initial triangulation and BA.")

        # Prepare observed 2D points for BA residual function ( N_inliers, N_cameras, 2)
        points_2d_observed_for_ba = inlier_points_all_cams

        # Use the new DLT function with all initial poses and corresponding 2D points
        points3D_initial = triangulate_points_dlt(points_2d_observed_for_ba, initial_poses, KS)
        # points_2d_observed_for_ba has shape (N_inliers, N_cameras, 2)
        # initial_poses is list of pose dicts (N_cameras)
        # KS is list of K matrices (N_cameras)
    else: # Handle single camera case
        print("Only one camera, cannot perform triangulation or BA.")
        return


    # Filter initial points for NaN/Inf
    valid_initial_mask = np.all(np.isfinite(points3D_initial), axis=1)
    points3D_init_filtered = points3D_initial[valid_initial_mask]
    # IMPORTANT: Keep corresponding 2D points aligned with filtered 3D points
    points_2d_observed_filtered = points_2d_observed_for_ba[valid_initial_mask]

    if points3D_init_filtered.shape[0] < 3: # Need points for BA
        print("Error: Not enough valid points after initial triangulation for BA.")
        return

    print(f"Initial 3D points shape for BA: {points3D_init_filtered.shape}")

    # --- Bundle Adjustment Setup ---
    # Initial parameters: flatten poses (R as rvec, t) for cams 1..N-1, then flatten 3D points
    params0 = []
    for i in range(1, NUM_CAMERAS):
        R_init = initial_poses[i]["R"]
        t_init = initial_poses[i]["t"]
        rvec_init, _ = cv2.Rodrigues(R_init)
        params0.extend(rvec_init.flatten())
        params0.extend(t_init.flatten())
    params0.extend(points3D_init_filtered.flatten())
    params0 = np.array(params0)

    n_points_ba = points3D_init_filtered.shape[0]

    # --- Run Bundle Adjustment ---
    print("Running Bundle Adjustment...")
    res = least_squares(
        bundle_adjustment_residual,
        params0,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        # Pass Ks and Dists as lists correctly
        args=(NUM_CAMERAS, n_points_ba, points_2d_observed_filtered, KS, DISTROTION_COEFFICIENTS)
    )

    # --- Extract Optimized Results ---
    params_optimized = res.x
    num_pose_params_per_cam = 6 # 3 rvec + 3 tvec
    num_static_params = 0 # If optimizing intrinsics, this would change
    # Calculate end index for pose parameters
    end_pose_params_idx = num_static_params + num_pose_params_per_cam * (NUM_CAMERAS - 1)
    pose_params_opt = params_optimized[num_static_params : end_pose_params_idx]
    points3D_optimized = params_optimized[end_pose_params_idx:].reshape((n_points_ba, 3))

    optimized_poses = [{"R": np.eye(3), "t": np.zeros((3, 1))}] # Cam 0 is reference
    print("--- Optimized Poses ---")
    print("Camera 0: R=eye(3), t=[0,0,0]")
    for i in range(NUM_CAMERAS - 1):
        rvec_opt = pose_params_opt[i * num_pose_params_per_cam : i * num_pose_params_per_cam + 3]
        tvec_opt = pose_params_opt[i * num_pose_params_per_cam + 3 : (i + 1) * num_pose_params_per_cam]
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        optimized_poses.append({"R": R_opt, "t": tvec_opt.reshape(3, 1)})
        print(f"Camera {i+1}:")
        print(f"  R:\n{R_opt}")
        print(f"  t:\n{tvec_opt.reshape(3, 1)}")
    
    # Save optimized poses to a JSON file
    poses_dict = {}
    for i, pose in enumerate(optimized_poses):
        poses_dict[f"camera_{i}"] = {
            "R": pose["R"].tolist(),
            "t": pose["t"].tolist()
        }
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimized_poses_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(poses_dict, f, indent=4)
    print(f"Optimized poses saved to {filename}")


    print("Bundle Adjustment finished.")
    print('Optimized 3D points:\n', points3D_optimized)

    # Plot the *optimized* 3D points and camera poses
    print(f'Plotting {points3D_optimized.shape[0]} optimized 3D points.')
    # Assuming plot_3d_points can take poses (update utils.py if needed)
    plot_3d_points(points3D_optimized, optimized_poses)


if __name__ == '__main__':
    main()

