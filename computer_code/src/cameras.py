from pseyepy import Camera
import cv2
import time
import numpy as np
from scipy import linalg
import json
from datetime import datetime
import logging
import typing as tp
from pathlib import Path
import itertools
from scipy.optimize import least_squares
from drone_tracker import DroneTracker

class Cameras:
    """
    A class to manage multiple ps3eye cameras.
    """

    FPS = 60
    THRESHOLD_VALUE = 3
    EXPOSURE = 180
    GAIN = 35
    RESOLUTION = Camera.RES_SMALL
    MIN_CALIBRATION_POINTS = 8
    POSES_DIR = Path(__file__).parent.parent / "poses"
    POSES_DIR.mkdir(parents=True, exist_ok=True)

    KERNEL = np.array([[-2,-1,-1,-1,-2],
                        [-1,1,3,1,-1],
                        [-1,3,4,3,-1],
                        [-1,1,3,1,-1],
                        [-2,-1,-1,-1,-2]])


    def __init__(self, num_cameras, camera_params_path="camera-params.json"):
        """
        Initialize the Cameras object.

        Args:
            num_cameras: number of cameras to initialize.
            camera_params_path: path to the JSON file containing the camera parameters.
        """
        self.num_cameras = num_cameras
        try:
            self.cameras = Camera(fps=self.FPS, resolution=self.RESOLUTION, gain=self.GAIN, exposure=self.EXPOSURE, colour=True)
        except Exception as e:
            logging.error(f"Cannot initialize cameras: {e}")
            self.cameras = None

        # Load camera parameters
        f = open(camera_params_path)
        camera_params = json.load(f)
        self.ks = [np.array(camera_params[i]["intrinsic_matrix"]) for i in range(self.num_cameras)]
        self.distortion_coefficients = [np.array(camera_params[i]["distortion_coef"]) for i in range(self.num_cameras)]
        self.rotations = [np.array(camera_params[i]["rotation"]) for i in range(self.num_cameras)]

        # Camera poses
        self.Rs = []
        self.ts = []
        
        # Initialize drone tracking system
        self.drone_tracker = DroneTracker(
            max_drones=4,
            max_association_distance=1.5,  
            track_timeout=5.0,  # 5 seconds
            cutoff_frequency=15.0,  # 15Hz low-pass filter
            sampling_frequency=self.FPS  # Use camera FPS as sampling frequency
        )
        
        # Validate loaded intrinsics
        self._validate_intrinsics()

    
    def load_poses(self, poses_path):
        """
        Load poses from a JSON file and rotate the camera rig so that the plane
        formed by the first 3 cameras is aligned with the X-Z plane.

        Args:
            poses_path: path to the JSON file containing the poses.
        """
        with open(poses_path) as f:
            poses = json.load(f)

        self.Rs = [np.array(poses[f"camera_{i}"]["R"]) for i in range(self.num_cameras)]
        self.ts = [np.array(poses[f"camera_{i}"]["t"]) for i in range(self.num_cameras)]
        # self.rotate_cameras()
        

    def rotate_cameras(self):
        """
        Rotate the cameras so that the plane formed by the first 3 cameras is aligned with the X-Z plane,
        and ensure they are looking down.
        """
        # Load initial Rs and ts 
        initial_Rs = self.Rs.copy()
        initial_ts = self.ts.copy()

        # Check if we have at least 3 cameras
        if self.num_cameras < 3:
            print("Not enough cameras to form a plane. Using original poses.")
            self.Rs = initial_Rs
            self.ts = initial_ts
            return

        # Get the positions of the first 3 cameras
        cam_positions = []
        for i in range(3):
            R_i = initial_Rs[i]
            t_i = initial_ts[i]
            pos_i = -R_i.T @ t_i
            cam_positions.append(pos_i.flatten())
        
        cam_positions = np.array(cam_positions)
        
        # Calculate two vectors on the camera plane
        v1 = cam_positions[1] - cam_positions[0]
        v2 = cam_positions[2] - cam_positions[0]
        
        # Calculate the normal vector of the camera plane
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # The target normal is [0, 1, 0] (Y-axis) for X-Z plane alignment
        target_normal = np.array([0, 1, 0])
        
        # Calculate the rotation axis and angle to align the normals
        rotation_axis = np.cross(normal, target_normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) if np.linalg.norm(rotation_axis) > 1e-6 else np.array([1, 0, 0])
        
        cos_angle = np.dot(normal, target_normal)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle = np.arccos(cos_angle)
        
        # Create the rotation matrix using Rodrigues formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Check if the cameras are looking up and apply a 180-degree rotation if needed
        # Assume the camera looks along the negative Y-axis in its local frame
        view_vector = np.array([0, -1, 0])
        transformed_view_vector = R_align @ view_vector

        if transformed_view_vector[1] > 0:  # If looking up
            R_flip = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            R_align = R_flip @ R_align
        
        # Apply the rotation to all cameras
        self.Rs = []
        self.ts = []
        for i in range(self.num_cameras):
            R_i_old = initial_Rs[i]
            t_i_old = initial_ts[i]
            
            # Apply the alignment rotation
            R_i_new = R_align @ R_i_old
            t_i_new = R_align @ t_i_old
            
            self.Rs.append(R_i_new)
            self.ts.append(t_i_new)


    def set_poses(self, poses):
        """
        Set the poses of the cameras.

        Args:
            poses: list of dictionaries with 'R' and 't' for each camera or
                dictionary with camera_0, camera_1, etc. as keys
        
        Note: Automatically applies rotate_cameras() to ensure cameras look down
        and are aligned in a plane perpendicular to the ground.
        """
        if isinstance(poses, list):
            # Handle list format (from calibrate_cameras)
            self.Rs = [np.array(pose["R"]) for pose in poses]
            self.ts = [np.array(pose["t"]) for pose in poses]
        else:
            # Handle dictionary format (from load_poses)
            self.Rs = [np.array(poses[f"camera_{i}"]["R"]) for i in range(self.num_cameras)]
            self.ts = [np.array(poses[f"camera_{i}"]["t"]) for i in range(self.num_cameras)]
        
        # Automatically correct camera orientation: align camera plane with X-Z plane and ensure cameras look down
        # self.rotate_cameras()


    def save_poses(self):
        """
        Save the poses of the cameras to a JSON file.
        Args:
            poses_name: name of the poses file.
        """
        poses_dict = {}
        for i in range(self.num_cameras):
            poses_dict[f"camera_{i}"] = {
                "R": self.Rs[i].tolist(),
                "t": self.ts[i].tolist()
            }

        poses_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_poses.json"

        with open(self.POSES_DIR / poses_name, 'w') as f:
            json.dump(poses_dict, f)

    
    def read_one_frame(self) -> list[np.ndarray]:
        """
        Read one frame from all cameras. 
        Returns:
            points_2d_observed: list of numpy arrays of 2D points, shape (num_cameras, [num_points_i, 2])
        """
        frames, _ = self.cameras.read()
        points_2d_observed = []
        for i, frame in enumerate(frames):
            frame = Cameras._preprocess_frame(frame, self.rotations[i], self.ks[i], self.distortion_coefficients[i])

            # Find dots
            frame, image_points = Cameras._find_dots(frame) # frame: [H, W, 3], image_points: [num_points, 2] (float32)
            points_2d_observed.append(image_points)

        return points_2d_observed
    

    def collect_calibration_points(self) -> tuple[np.ndarray, tp.Optional[np.ndarray]]:
        """
        LEGACY METHOD: Collect calibration points from all cameras (single point per camera).
        Returns:
            frames: numpy array of frames. (num_cameras, H, W, 3)
            points: numpy array of image points if exactly one point is detected in each camera (num_cameras, 2), 
                    otherwise an array containing np.nan where points are missing or if multiple points detected.
        """
        frames, _ = self.cameras.read()
        points = np.full((self.num_cameras, 2), np.nan, dtype=np.float32) # Initialize with nans
        processed_frames = []

        for i, frame in enumerate(frames):
            processed_frame = Cameras._preprocess_frame(frame, self.rotations[i], self.ks[i], self.distortion_coefficients[i])
            # Find dots
            viz_frame, image_points = Cameras._find_dots(processed_frame) # image_points is np.array, dtype=float32
            processed_frames.append(viz_frame)

            if image_points.shape[0] > 1:
                logging.debug(f"More than one point detected in camera {i}.")
                # points[i] remains np.nan, np.nan
            elif image_points.shape[0] == 0 or np.isnan(image_points[0, 0]): # No points or was [[np.nan, np.nan]]
                logging.debug(f"No points detected in camera {i}.")
                # points[i] remains np.nan, np.nan
            else:
                points[i] = image_points[0] # Assign the first (and only) point
        
        # Check if any camera failed to detect a single point
        if np.any(np.isnan(points)):
            return np.array(processed_frames), None # Return None for points if any camera failed
        
        # Enhanced validation for stream calibration
        if not self._validate_calibration_frame_quality(points):
            return np.array(processed_frames), None

        return np.array(processed_frames), points
    
    def collect_single_point_frame(self) -> tuple[np.ndarray, tp.Optional[np.ndarray]]:
        """
        SIMPLIFIED METHOD: Collect frames where each camera detects exactly one point.
        Points are matched by temporal synchronization - no correspondence calculation needed.
        
        This is the preferred method for stream calibration as it's faster and more reliable
        than the multi-blob epipolar correspondence approach.
        
        Returns:
            frames: numpy array of frames. (num_cameras, H, W, 3)
            points: numpy array of image points if exactly one point is detected in each camera (num_cameras, 2), 
                    otherwise None if any camera has 0 or >1 points.
        """
        try:
            if self.cameras is None:
                logging.debug("Camera object is None")
                return np.array([]), None
                
            frames, _ = self.cameras.read()
            if frames is None or len(frames) == 0:
                logging.debug("No frames received from cameras")
                return np.array([]), None
                
            points = np.full((self.num_cameras, 2), np.nan, dtype=np.float32)
            processed_frames = []
            
            # Process each camera frame
            for i, frame in enumerate(frames):
                if frame is None:
                    logging.debug(f"No frame received from camera {i}")
                    return np.array([]), None  # Fail fast if any camera missing
                    
                try:
                    processed_frame = Cameras._preprocess_frame(frame, self.rotations[i], self.ks[i], self.distortion_coefficients[i])
                    viz_frame, image_points = Cameras._find_dots(processed_frame)
                    processed_frames.append(viz_frame)
                    
                    # Check for exactly one point
                    if image_points.shape[0] == 1 and not np.isnan(image_points[0, 0]):
                        points[i] = image_points[0]
                    else:
                        # Not exactly one valid point - reject this frame
                        logging.debug(f"Camera {i}: detected {image_points.shape[0]} points (need exactly 1)")
                        return np.array(processed_frames), None
                        
                except Exception as e:
                    logging.debug(f"Error processing frame from camera {i}: {e}")
                    return np.array([]), None
            
            # All cameras have exactly one point - validate quality
            if not self._validate_calibration_frame_quality(points):
                logging.debug("Frame quality validation failed")
                return np.array(processed_frames), None
            
            logging.debug(f"Successfully collected single-point frame from all {self.num_cameras} cameras")
            return np.array(processed_frames), points
            
        except Exception as e:
            logging.error(f"Error in collect_single_point_frame: {e}")
            return np.array([]), None

    def capture_points(self) -> np.ndarray:
        """
        Capture points from all cameras. For each camera, only one point can be captured.
        Returns:
            points_2d_observed: numpy array of image points. (num_points_captured, num_cameras, 2)
        """
        
        list_of_points_per_camera = [] # Will be list of (num_cameras, 2) arrays
        print("Press Enter to capture points | Press q to quit")
        while True:
            frames, detected_points = self.collect_calibration_points() # detected_points is (num_cameras, 2) or None

            # Display frames
            if frames is not None and len(frames) > 0:
                combined_frame = np.hstack(frames[:self.num_cameras])
                cv2.imshow('PS3i Cameras - Detected Points', combined_frame)

            key = cv2.waitKey(1) & 0xFF
            # Capture points
            if key == 13:  # 13 is the keycode for Enter
                if detected_points is None: # This means collect_calibration_points decided it's not a valid set
                    print("Error: Not all cameras detected a single valid point.")
                else:
                    # detected_points is already (num_cameras, 2)
                    list_of_points_per_camera.append(detected_points)
                    print(f"Captured point set {len(list_of_points_per_camera)}. Press Enter to capture next points | Press q to quit")
            
            # Quit
            if key == ord('q'):
                break
            
            time.sleep(1/self.FPS) # Small delay
        
        cv2.destroyAllWindows()

        if not list_of_points_per_camera:
            return np.empty((0, self.num_cameras, 2), dtype=np.float32)
            
        # Convert list of (num_cameras, 2) arrays to (num_captured_sets, num_cameras, 2)
        return np.array(list_of_points_per_camera, dtype=np.float32)
    
    def _validate_calibration_frame_quality(self, points: np.ndarray) -> bool:
        """
        Enhanced validation for calibration frame quality.
        
        Args:
            points: Array of shape (num_cameras, 2) with detected points
            
        Returns:
            bool: True if frame meets quality criteria
        """
        if points.shape[0] != self.num_cameras or points.shape[1] != 2:
            return False
        
        # Check if any points are NaN
        if np.any(np.isnan(points)):
            return False
        
        # Check if points are within reasonable image bounds (PS3 Eye cameras: 320x240)
        for i, point in enumerate(points):
            x, y = point
            # PS3 Eye camera resolution bounds with edge margin
            if x < 20 or x > 300 or y < 20 or y > 220:  # Keep points away from edges
                logging.debug(f"Camera {i} point too close to image edge: ({x:.1f}, {y:.1f})")
                return False
        
        # Check point distribution - ensure points aren't too clustered
        if len(points) > 1:
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    # Calculate distance between points in different cameras
                    # This prevents having very similar viewpoints
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist < 30:  # Minimum pixel distance between cameras' views
                        logging.debug(f"Points too close between cameras {i} and {j}: distance {dist:.1f}")
                        return False
        
        return True
    
    def stream_calibration_points(self, min_points: int = 200, max_duration: float = 120.0) -> list[np.ndarray]:
        """
        Collect calibration points from continuous video stream.
        
        Args:
            min_points: Minimum number of points to collect
            max_duration: Maximum collection time in seconds
            
        Returns:
            List of calibration point arrays, each with shape (num_cameras, 2)
        """
        collected_points = []
        start_time = time.time()
        frames_processed = 0
        
        logging.info(f"Starting stream calibration collection (target: {min_points} points, max time: {max_duration}s)")
        
        while len(collected_points) < min_points and (time.time() - start_time) < max_duration:
            frames, points = self.collect_calibration_points()
            frames_processed += 1
            
            if points is not None:
                collected_points.append(points)
                
                # Log progress periodically
                if len(collected_points) % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = len(collected_points) / elapsed
                    logging.info(f"Collected {len(collected_points)} points in {elapsed:.1f}s (rate: {rate:.1f} pts/s)")
            
            # Small delay to prevent overwhelming the system
            time.sleep(1/60)  # 60 FPS max collection rate
        
        duration = time.time() - start_time
        success_rate = len(collected_points) / frames_processed if frames_processed > 0 else 0
        
        logging.info(f"Stream calibration complete: {len(collected_points)} points collected in {duration:.1f}s")
        logging.info(f"Frame success rate: {success_rate:.1%} ({len(collected_points)}/{frames_processed})")
        
        return collected_points


    def estimate_relative_pose(self, image_points) -> tuple[list[dict], np.ndarray]:
        """
        Estimate the relative pose using cv2.recoverPose and return inlier indices.

        Args:
            image_points: numpy array of image points. (num_points, num_cameras, 2)

        Returns:
            List of camera poses (dictionaries with 'R' and 't'), relative to camera 0.
            Indices of the inlier points used for the Camera 0 <-> Camera 1 pose estimation.
            Returns None, None if pose estimation fails for any required pair.
        """

        if image_points.shape[0] < 5 or image_points.shape[1] < self.num_cameras:
            logging.error(f"Error: Need at least 5 points and {self.num_cameras} camera views. Got shape {image_points.shape}")
            return None, None

        # Initialize list to store poses, camera 0 is the reference
        camera_poses = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
        inlier_indices_01 = None # Store inliers for cam0-cam1 pair

        points_cam0 = image_points[:, 0, :].astype(np.float32) # Points in reference camera
        K0 = self.ks[0] # Intrinsics for reference camera

        for i in range(1, self.num_cameras):
            logging.info(f"--- Estimating Pose for Camera {i} relative to Camera 0 ---")
            points_cami = image_points[:, i, :].astype(np.float32)
            original_indices = np.arange(image_points.shape[0])

            # --- Find Fundamental Matrix ---
            F, mask_fundamental = cv2.findFundamentalMat(
                points_cam0, points_cami, 
                method=cv2.FM_RANSAC, 
                ransacReprojThreshold=1.0,  # Tighter threshold
                confidence=0.99,
                maxIters=2000
            )

            if F is None:
                logging.error(f"Error: Could not compute Fundamental Matrix between Cam 0 and Cam {i}.")
                return None, None

            fundamental_mask_bool = mask_fundamental.ravel() == 1
            inliers0 = points_cam0[fundamental_mask_bool]
            inliersi = points_cami[fundamental_mask_bool]
            indices_after_fundamental = original_indices[fundamental_mask_bool]

            logging.info(f"Fundamental matrix: {len(inliers0)}/{len(points_cam0)} points passed for Cam 0 <-> Cam {i}")
            
            if len(inliers0) < 8:  # Need more points for robust pose estimation
                logging.error(f"Error: Not enough inliers ({len(inliers0)}) after F matrix calculation between Cam 0 and Cam {i}.")
                return None, None

            # --- Use Reference Implementation Approach ---
            Ki = self.ks[i] # Intrinsics for camera i
            
            # E = K1^T * F * K0 
            E = Ki.T @ F @ K0
            
            # Get the best pose solution using standard OpenCV
            retval, R_candidates, t_candidates, mask_recover = cv2.recoverPose(
                E, inliers0, inliersi, K0
            )
            
            if R_candidates is None or t_candidates is None:
                logging.error(f"Error: cv2.recoverPose failed for Cam 0 <-> Cam {i}")
                return None, None
            
            # Use the mask from recoverPose if available, otherwise use fundamental matrix inliers
            if mask_recover is not None:
                # In the mask returned by cv2.recoverPose the inlier flag can be either 1 or 255.
                # Treat any non-zero entry as an inlier to avoid accidentally discarding valid points.
                final_mask = mask_recover.ravel() != 0
                final_indices = indices_after_fundamental[final_mask]
                num_valid_points = np.sum(final_mask)
            else:
                # If no mask from recoverPose, use all fundamental matrix inliers
                final_indices = indices_after_fundamental
                num_valid_points = len(indices_after_fundamental)
            
            R = R_candidates
            t = t_candidates.reshape(3, 1)

            logging.info(f"Cam 0 <-> Cam {i}: Pose recovered with {num_valid_points} valid points (retval: {retval})")
            
            # Validate pose quality
            translation_norm = np.linalg.norm(t)
            rotation_angle = np.linalg.norm(cv2.Rodrigues(R)[0])
            
            logging.info(f"Cam {i} pose - translation norm: {translation_norm:.3f}, rotation angle: {rotation_angle:.3f} rad")

            # Store the calculated pose for camera i
            camera_poses.append({"R": R, "t": t})

            # Store inliers for the first pair (camera 0-1)
            if i == 1:
                inlier_indices_01 = final_indices
                logging.info(f"Stored {len(inlier_indices_01)} inlier indices from Cam 0-1 pair.")

        if inlier_indices_01 is None and self.num_cameras > 1:
            logging.error("Error: Failed to determine inliers for the Cam 0-1 pair.")
            return None, None

        logging.info(f"Successfully estimated poses for all {self.num_cameras} cameras.")
        return camera_poses, inlier_indices_01
    

    def run_bundle_adjustment(self, points_2d_observed, initial_poses) -> tp.Optional[tuple[list[dict], np.ndarray, np.ndarray]]:
        """
        Run bundle adjustment on the observed points and initial poses.

        Args:
            points_2d_observed: numpy array of image points. (num_points, num_cameras, 2)
            initial_poses: list of dictionaries with 'R' and 't' for each camera.

        Returns:
            Optimized poses and 3D points.
        """
        Rs = [pose["R"] for pose in initial_poses]
        ts = [pose["t"] for pose in initial_poses]
        points_3d_initial = self._triangulate_points(points_2d_observed, Rs, ts, self.ks)

        # Filter initial points for NaN/Inf
        valid_initial_mask = np.all(np.isfinite(points_3d_initial), axis=1)
        points_2d_observed_filtered = points_2d_observed[valid_initial_mask]
        points_3d_initial_filtered = points_3d_initial[valid_initial_mask]

        if points_3d_initial_filtered.shape[0] < 8:  # Need more points for BA
            logging.error("Error: Not enough valid points after initial triangulation for BA.")
            return None
        
        logging.info(f"Starting bundle adjustment with {points_3d_initial_filtered.shape[0]} points")
        
        # Simple bundle adjustment approach (similar to reference)
        # Initial parameters: poses (R as rvec, t) for cams 1..N-1, then flatten 3D points
        params0 = []
        
        # Add poses for cameras 1..N-1 (camera 0 remains reference)
        for i in range(1, self.num_cameras):
            R_init = initial_poses[i]["R"]
            t_init = initial_poses[i]["t"]
            rvec_init, _ = cv2.Rodrigues(R_init)
            params0.extend(rvec_init.flatten())
            params0.extend(t_init.flatten())
        
        # Add 3D points
        params0.extend(points_3d_initial_filtered.flatten())
        params0 = np.array(params0)

        n_points_ba = points_3d_initial_filtered.shape[0]

        # Run bundle adjustment with simpler approach
        logging.info("Running Bundle Adjustment...")
        # Use zero-distortion coefficients because input 2-D points are already undistorted
        zero_distortion = [np.zeros_like(d) for d in self.distortion_coefficients]
        res = least_squares(
            self._bundle_adjustment_residual,
            params0,
            verbose=1,
            x_scale='jac',
            ftol=1e-4,
            method='trf',
            max_nfev=1000,
            args=(self.num_cameras, n_points_ba, points_2d_observed_filtered, self.ks, zero_distortion)
        )

        if not res.success:
            logging.warning(f"Bundle adjustment did not converge: {res.message}")
        
        # Extract Optimized Results
        params_optimized = res.x
        
        # Extract pose parameters
        num_pose_params_per_cam = 6 # 3 rvec + 3 tvec
        pose_start_idx = 0
        pose_end_idx = pose_start_idx + num_pose_params_per_cam * (self.num_cameras - 1)
        pose_params_opt = params_optimized[pose_start_idx:pose_end_idx]
        
        # Extract 3D points
        points3D_optimized = params_optimized[pose_end_idx:].reshape((n_points_ba, 3))

        # Extract optimized poses
        optimized_poses = [{"R": np.eye(3), "t": np.zeros((3, 1))}] # Cam 0 is reference
        for i in range(self.num_cameras - 1):
            rvec_opt = pose_params_opt[i * num_pose_params_per_cam : i * num_pose_params_per_cam + 3]
            tvec_opt = pose_params_opt[i * num_pose_params_per_cam + 3 : (i + 1) * num_pose_params_per_cam]
            R_opt, _ = cv2.Rodrigues(rvec_opt)
            optimized_poses.append({"R": R_opt, "t": tvec_opt.reshape(3, 1)})

        # Calculate final reprojection errors
        final_errors = []
        for cam_idx in range(self.num_cameras):
            for pt_idx in range(n_points_ba):
                observed_2d = points_2d_observed_filtered[pt_idx, cam_idx, :]
                if np.all(np.isfinite(observed_2d)):
                    projected_2d = self._project(
                        points3D_optimized[pt_idx:pt_idx+1], 
                        self.ks[cam_idx], 
                        optimized_poses[cam_idx]["R"], 
                        optimized_poses[cam_idx]["t"], 
                        np.zeros(5)  # Points are in undistorted space
                    )
                    if projected_2d.shape[0] > 0:
                        error = np.linalg.norm(projected_2d[0] - observed_2d)
                        final_errors.append(error)

        mean_error = np.mean(final_errors) if final_errors else float('inf')
        logging.info(f"Bundle adjustment completed. Mean reprojection error: {mean_error:.2f} pixels")
        
        if mean_error > 10.0:
            logging.warning(f"High reprojection error after bundle adjustment: {mean_error:.2f} pixels")

        return optimized_poses, points3D_optimized, points_2d_observed_filtered

    @staticmethod
    def _bundle_adjustment_residual(params, n_cameras, n_points, points_2d_observed, Ks, Dists) -> np.ndarray:
        """
        Simplified bundle adjustment residual function (similar to reference implementation).
        
        Args:
            params: Flat array containing:
                    - R_vec_cam1..N-1 (3 * (n_cameras-1)): Rotation vectors for cameras 1 to N-1
                    - t_vec_cam1..N-1 (3 * (n_cameras-1)): Translation vectors for cameras 1 to N-1
                    - points_3d_flat (3 * n_points): Flattened 3D point coordinates
            n_cameras: Total number of cameras
            n_points: Number of 3D points
            points_2d_observed: Observed 2D points, shape (n_points, n_cameras, 2)
            Ks: List of camera intrinsic matrices
            Dists: List of distortion coefficients
            
        Returns:
            Flattened array of residuals (reprojection errors)
        """
        # --- Unpack Parameters ---
        num_pose_params_per_cam = 6
        pose_end_idx = num_pose_params_per_cam * (n_cameras - 1)
        pose_params = params[:pose_end_idx]
        
        # Extract 3D points
        points_3d = params[pose_end_idx:].reshape((n_points, 3))

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
            dist = Dists[cam_idx]
            R = camera_poses[cam_idx]["R"]
            t = camera_poses[cam_idx]["t"]

            # Project the current estimate of 3D points into this camera
            points_2d_proj = Cameras._project(points_3d, K, R, t, dist)
            # Get the corresponding observed 2D points
            observed = points_2d_observed[:, cam_idx, :] # Shape (N, 2)

            # Handle potential missing observations
            valid_obs_mask = np.all(np.isfinite(observed), axis=1)
            if not np.all(valid_obs_mask):
                observed = observed[valid_obs_mask]
                points_2d_proj = points_2d_proj[valid_obs_mask]
                if observed.shape[0] == 0: 
                    continue # Skip camera if no valid points left

            # Calculate error: observed - projected
            errors = observed - points_2d_proj # Shape (N_valid, 2)
            all_residuals.append(errors) # Add errors for this camera

        # Flatten all valid residuals into a single 1D array
        flat_residuals = np.concatenate([res.flatten() for res in all_residuals])
        return flat_residuals

    def triangulate_points(self, points_2d_observed) -> np.ndarray:
        """
        Triangulate points from all cameras.

        Args:
            points_2d_observed: numpy array of image points. (num_points, num_cameras, 2)

        Returns:
            numpy array of 3D points. (num_points, 3)
        """
        return Cameras._triangulate_points(points_2d_observed, self.Rs, self.ts, self.ks)


    def stream_triangulated_points(self) -> tp.Generator[np.ndarray, None, None]:
        """
        Streams triangulated points.
        Yields:
            numpy array of 3D points. (num_points, 3)
        """
        while True:
            points_2d_observed = self.read_one_frame()
            correspondences = self.find_epipolar_correspondences(points_2d_observed) 
            points_3d = self.triangulate_points(correspondences)
            yield points_3d


    def stream_triangulated_points_with_drones(self) -> tp.Generator[np.ndarray, None, None]:
        """
        Streams triangulated points and detects drones.
        """
        while True:
            points_2d_observed = self.read_one_frame()
            correspondences = self.find_epipolar_correspondences(points_2d_observed) 
            points_3d = self.triangulate_points(correspondences)
            drones = self.find_drones(points_3d)
            yield points_3d, drones

    def stream_filtered_drones(self) -> tp.Generator[tp.List[dict], None, None]:
        """
        Stream tracked and filtered drone data using Kalman filtering and low-pass filtering.
        
        Yields:
            List of filtered drone state dictionaries with keys:
            - 'position': [x, y, z] filtered position
            - 'velocity': [vx, vy, vz] filtered velocity 
            - 'acceleration': [ax, ay, az] estimated acceleration
            - 'heading': filtered heading angle
            - 'track_id': persistent track identifier
            - 'confidence': tracking confidence [0-1]
            - 'age': track age in seconds
        """
        logging.info("Starting filtered drone streaming")
        
        while True:
            try:
                # Get raw detections
                points_2d_observed = self.read_one_frame()
                correspondences = self.find_epipolar_correspondences(points_2d_observed) 
                points_3d = self.triangulate_points(correspondences)
                raw_drones = self.find_drones(points_3d)
                
                # Apply tracking and filtering
                tracked_drones = self.drone_tracker.update(raw_drones)
                
                yield tracked_drones, points_3d
                
            except Exception as e:
                logging.error(f"Error in stream_filtered_drones: {e}")
                yield []  # Return empty list on error
                time.sleep(1/self.FPS)  # Continue at regular interval


    def find_epipolar_correspondences(self, points_2d_observed: list[np.ndarray], max_dist=5.0) -> np.ndarray:
        """
        For each point in camera 0, find the best matching point in each other camera using epipolar geometry.
        Args:
            points_2d_observed: list of numpy arrays of 2D points, one array per camera.
                                Each array has shape [num_points_i, 2], dtype=float32.
            max_dist: maximum pixel distance from epipolar line to consider a match
        Returns:
            correspondences: numpy array of shape (num_points_in_cam0, num_cameras, 2), 
                             where each element is a tuple of (x, y) or np.nan if no match. dtype=float32.
        """
        num_cameras = len(points_2d_observed)
        if num_cameras == 0 or points_2d_observed[0].shape[0] == 0:
            return np.empty((0, num_cameras, 2), dtype=np.float32)

        correspondences = []

        # For each point in camera 0
        for pt0_idx in range(points_2d_observed[0].shape[0]):
            pt0 = points_2d_observed[0][pt0_idx, :] # This is a (2,) array

            if np.isnan(pt0[0]): # Skip if the point in cam0 is already NaN
                match = [np.array([np.nan, np.nan])] * num_cameras # Fill with NaNs for all cams
                correspondences.append(match)
                continue

            match = [pt0]  # Start with the point in cam0
            for cam_idx in range(1, num_cameras):
                pts_i = points_2d_observed[cam_idx] # This is a (num_points_cam_i, 2) array
                
                if pts_i.shape[0] == 0: # No points in current camera
                    match.append(np.array([np.nan, np.nan]))
                    continue

                # Compute Fundamental matrix between cam0 and cam_idx
                K0, Ki = self.ks[0], self.ks[cam_idx]
                R0, Ri = self.Rs[0], self.Rs[cam_idx]
                t0, ti = self.ts[0], self.ts[cam_idx]

                # Relative pose from 0 to i
                R_rel = Ri @ R0.T
                t_rel = ti - R_rel @ t0

                # Skew-symmetric matrix for t_rel
                tx = np.array([
                    [0, -t_rel[2][0], t_rel[1][0]],
                    [t_rel[2][0], 0, -t_rel[0][0]],
                    [-t_rel[1][0], t_rel[0][0], 0]
                ])
                E = tx @ R_rel
                F = np.linalg.inv(Ki).T @ E @ np.linalg.inv(K0)

                # Compute epipolar line in cam_idx for pt0
                pt0_h = np.array([pt0[0], pt0[1], 1.0], dtype=np.float32)
                line = F @ pt0_h  # ax + by + c = 0

                # Find the closest point in cam_idx to the epipolar line
                min_dist_sq = float('inf') # Using squared distance to avoid sqrt
                best_pt = np.array([np.nan, np.nan])

                for pt_idx in range(pts_i.shape[0]):
                    pt = pts_i[pt_idx, :]
                    if np.isnan(pt[0]): # Skip if current point is NaN
                        continue
                    
                    pt_h = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
                    # dist = abs(line @ pt_h) / np.sqrt(line[0]**2 + line[1]**2) # distance point to line
                    # Simplified distance check: perpendicular distance squared
                    # dist_val_numerator = (line[0]*pt[0] + line[1]*pt[1] + line[2])**2
                    # dist_val_denominator = (line[0]**2 + line[1]**2)
                    
                    # Robust check for denominator
                    # if dist_val_denominator < 1e-9: # line is degenerate (e.g. [0,0,c])
                    #     continue 
                    # dist_sq = dist_val_numerator / dist_val_denominator
                    
                    # Epipolar constraint: pt_i.T @ F @ pt0 = 0
                    # For a given pt0, and a candidate pt_i, the epipolar line for pt0 in image i is l_i = F @ pt0_h
                    # The distance from pt_i to l_i is (pt_i_h.T @ l_i) / sqrt(l_i[0]^2 + l_i[1]^2)
                    
                    numerator = pt_h @ line # This is a scalar
                    denominator_sq = line[0]**2 + line[1]**2

                    if denominator_sq < 1e-9: # Avoid division by zero if line is degenerate
                        dist_sq = float('inf')
                    else:
                        dist_sq = (numerator**2) / denominator_sq


                    if dist_sq < min_dist_sq and dist_sq < (max_dist**2):
                        min_dist_sq = dist_sq
                        best_pt = pt
                match.append(best_pt)
            correspondences.append(match)

        # Convert list of lists of (2,) arrays to a single 3D numpy array
        # correspondences is currently a list of "match" lists. Each "match" is a list of (2,) arrays.
        # Desired shape: (num_points_in_cam0, num_cameras, 2)
        if not correspondences:
             return np.empty((0, num_cameras, 2), dtype=np.float32)
        
        return np.array(correspondences, dtype=np.float32)
    

    def scale_poses_from_drone_geometry(self,
                                       p_base_led1: np.ndarray,
                                       p_base_led2: np.ndarray,
                                       p_offset_led: np.ndarray,
                                       known_dist_base: float = 0.15, # meters
                                       known_dist_offset_to_midpoint: float = 0.05 # meters
                                       ) -> tp.Optional[float]:
        """
        Scales the camera translation vectors (self.ts) using known distances
        between three specific 3D points, assumed to be from a drone's LEDs.

        The method assumes:
        - p_base_led1 and p_base_led2 are a pair of LEDs with a known distance.
        - p_offset_led is a third LED with a known distance to the midpoint of p_base_led1 and p_base_led2.

        Args:
            p_base_led1: 3D coordinates of the first LED of the base pair in the current system's scale.
            p_base_led2: 3D coordinates of the second LED of the base pair in the current system's scale.
            p_offset_led: 3D coordinates of the offset LED in the current system's scale.
            known_dist_base: Known metric distance between p_base_led1 and p_base_led2 (e.g., 0.15m).
            known_dist_offset_to_midpoint: Known metric distance from p_offset_led
                                           to the midpoint of p_base_led1 and p_base_led2 (e.g., 0.10m).

        Returns:
            The calculated average scale factor if successful, otherwise None.
            This factor should also be used to scale any 3D points triangulated
            using the old (unscaled) poses.
            Modifies self.ts in place.
        """
        p_base_led1 = np.asarray(p_base_led1).flatten()
        p_base_led2 = np.asarray(p_base_led2).flatten()
        p_offset_led = np.asarray(p_offset_led).flatten()

        if not (p_base_led1.shape == (3,) and p_base_led2.shape == (3,) and p_offset_led.shape == (3,)):
            logging.error("Error: Input LED coordinates must be 3D points.")
            return None

        # Calculate current distance for the base pair in the arbitrary scale
        current_dist_base = np.linalg.norm(p_base_led2 - p_base_led1)
        if current_dist_base < 1e-6:
            logging.error("Error: Current distance between base LEDs is too small for reliable scaling.")
            return None
        
        scale_factor_base = known_dist_base / current_dist_base

        # Calculate midpoint of the current base pair
        current_midpoint_base = (p_base_led1 + p_base_led2) / 2.0
        
        # Calculate current distance from offset LED to midpoint in the arbitrary scale
        current_dist_offset = np.linalg.norm(p_offset_led - current_midpoint_base)
        if current_dist_offset < 1e-6:
            logging.error("Error: Current distance from offset LED to midpoint is too small for reliable scaling.")
            return None

        scale_factor_offset = known_dist_offset_to_midpoint / current_dist_offset
        
        # Check if the scale factors are consistent
        # A large discrepancy might indicate incorrect point identification or noisy 3D points.
        if abs(scale_factor_base - scale_factor_offset) / max(scale_factor_base, scale_factor_offset, 1e-6) > 0.25: # 25% relative difference
            logging.warning(
                f"Scale factors derived from base LED pair ({scale_factor_base:.4f}) and "
                f"offset LED ({scale_factor_offset:.4f}) differ significantly (>{abs(scale_factor_base - scale_factor_offset)/max(scale_factor_base, scale_factor_offset, 1e-6)*100:.1f}%). "
                "This might indicate incorrect LED identification or noisy 3D points. "
                "Proceeding with the average scale factor."
            )

        final_scale_factor = (scale_factor_base + scale_factor_offset) / 2.0
        
        if final_scale_factor < 1e-6 or not np.isfinite(final_scale_factor): 
            logging.error(f"Error: Calculated final scale factor ({final_scale_factor:.4f}) is invalid.")
            return None
        
        scaled_ts = []
        for t_vec in self.ts:
            scaled_ts.append(t_vec * final_scale_factor)
        self.ts = scaled_ts

        return final_scale_factor
    

    def try_scale_poses(self):
        """
        Try to scale the poses using the known distances between the LEDs.
        
        Returns:
            True if scaling was successful, False otherwise.
        """
        # Get points from cameras
        points_2d_observed = self.read_one_frame()
        correspondences = self.find_epipolar_correspondences(points_2d_observed) 
        points_3d = self.triangulate_points(correspondences)

        if points_3d is None or points_3d.shape[0] != 3:
            logging.error("Wrong number of points to scale poses.")
            return False
        
        # Determine which points are the base LEDs and the offset LED
        point_a = points_3d[0]
        point_b = points_3d[1]
        point_c = points_3d[2]

        # Calculate distances between points
        dist_ab = np.linalg.norm(point_a - point_b)
        dist_ac = np.linalg.norm(point_a - point_c)
        dist_bc = np.linalg.norm(point_b - point_c)

        # Longest distance is the base pair
        if dist_ab > dist_ac and dist_ab > dist_bc:
            p_base_led1 = point_a
            p_base_led2 = point_b
            p_offset_led = point_c
        elif dist_ac > dist_ab and dist_ac > dist_bc:
            p_base_led1 = point_a
            p_base_led2 = point_c
            p_offset_led = point_b
        else:
            p_base_led1 = point_b
            p_base_led2 = point_c
            p_offset_led = point_a

        scale = self.scale_poses_from_drone_geometry(p_base_led1, p_base_led2, p_offset_led)
        if scale is None:
            logging.error("Failed to scale poses.")
            return False
        
        return True

    def reset_drone_tracker(self):
        """Reset the drone tracking system."""
        self.drone_tracker.reset()
        logging.info("Drone tracker reset")

    def get_tracking_statistics(self) -> dict:
        """
        Get drone tracking performance statistics.
        
        Returns:
            Dictionary with tracking metrics
        """
        return self.drone_tracker.get_statistics()

    def configure_tracking(self, 
                          max_association_distance: float = None,
                          track_timeout: float = None,
                          cutoff_frequency: float = None):
        """
        Configure tracking parameters.
        
        Args:
            max_association_distance: Maximum distance for track association (meters)
            track_timeout: Track timeout in seconds
            cutoff_frequency: Low-pass filter cutoff frequency (Hz)
        """
        if max_association_distance is not None:
            self.drone_tracker.max_association_distance = max_association_distance
            logging.info(f"Updated max association distance to {max_association_distance}m")
            
        if track_timeout is not None:
            self.drone_tracker.track_timeout = track_timeout
            logging.info(f"Updated track timeout to {track_timeout}s")
            
        if cutoff_frequency is not None:
            self.drone_tracker.cutoff_frequency = cutoff_frequency
            # Note: Changing cutoff frequency only affects new tracks
            logging.info(f"Updated cutoff frequency to {cutoff_frequency}Hz (affects new tracks)")
        

    def set_ground(self, camera_height: float):
        """
        Adjust the cameras translation y value to match the ground height.
        """
        if not self.ts:
            logging.error("Poses (self.ts) are not loaded. Cannot set ground height.")
            return

        # Ensure camera_height is a float
        try:
            camera_height_float = float(camera_height)
        except (ValueError, TypeError):
            logging.error(f"Invalid ground_height value: {camera_height}. Must be a number.")
            return

        # Assuming self.ts[0] is a (3,1) numpy array like np.array([[x],[y],[z]])
        # Access the scalar y-value using self.ts[0][1, 0]
        current_y_offset = self.ts[0][1, 0] 
        diff = camera_height_float - current_y_offset
        logging.info(f"Setting ground height to {camera_height_float} meters. Current Y: {current_y_offset}. Diff: {diff} meters.")

        adjustment_vector = np.array([[0.0], [diff], [0.0]])
        self.ts = [t + adjustment_vector for t in self.ts]
    
    def calibrate_cameras(self, calibration_points) -> tuple[tp.List[dict], np.ndarray]:
        """
        Calibrate the intrinsic and extrinsic parameters of the cameras. Run bundle adjustment on the observed points and initial poses.
        Args:
            calibration_points: numpy array of 2d points. (num_points, num_cameras, 2)
        Returns:
            optimized_poses: list of dictionaries with 'R' and 't' for each camera.
            points3D_optimized: numpy array of 3D points. (num_points, 3)
        """
        if calibration_points is None or type(calibration_points) is not np.ndarray:
            logging.error("No calibration points provided.")
            return
        
        if calibration_points.shape[0] < self.MIN_CALIBRATION_POINTS:
            logging.error("Not enough calibration points received.")
            return
        
        if calibration_points.shape[1] != self.num_cameras:
            logging.error("Calibration points do not match number of cameras.")
            return
        
        if calibration_points.shape[2] != 2:
            logging.error("Calibration points must be 2D.")
            return
        
        # Estimate relative poses
        initial_poses, inlier_indices_01 = self.estimate_relative_pose(calibration_points)

        if initial_poses is None:
            logging.error("Error: Failed to estimate relative poses.")
            return
        # Check if we got inliers for the crucial 0-1 pair
        if self.num_cameras > 1 and inlier_indices_01 is None:
            logging.error("Exiting because pose estimation failed to get inliers for Cam 0-1 pair.")
            return
        if self.num_cameras > 1 and len(inlier_indices_01) == 0:
            logging.error("Exiting because pose estimation returned 0 inliers for Cam 0-1 pair.")
            return
        
        # Initial triangulation
        if self.num_cameras > 1:
            # Select only the inlier points (rows) across *all* cameras
            inlier_points_all_cams = calibration_points[inlier_indices_01, :, :]
            n_inliers = inlier_points_all_cams.shape[0]
            print(f"n_inliers: {n_inliers}")
            # Prepare observed 2D points for BA residual function ( N_inliers, N_cameras, 2)
            points_2d_observed_for_ba = inlier_points_all_cams

            # Use the new DLT function with all initial poses and corresponding 2D points
            Rs = [pose["R"] for pose in initial_poses]
            ts = [pose["t"] for pose in initial_poses]
            points3D_initial = Cameras._triangulate_points(points_2d_observed_for_ba, Rs, ts, self.ks)
            # points_2d_observed_for_ba has shape (N_inliers, N_cameras, 2)
            # initial_poses is list of pose dicts (N_cameras)
            # KS is list of K matrices (N_cameras)
        else: # Handle single camera case
            logging.error("Only one camera, cannot perform triangulation or BA.")
            return
        
        # Filter initial points for NaN/Inf
        valid_initial_mask = np.all(np.isfinite(points3D_initial), axis=1)
        points3D_init_filtered = points3D_initial[valid_initial_mask]
        points_2d_observed_filtered = points_2d_observed_for_ba[valid_initial_mask]
        print(f"points3D_init_filtered.shape: {points3D_init_filtered.shape}")
        print(f"points_2d_observed_filtered.shape: {points_2d_observed_filtered.shape}")

        if points3D_init_filtered.shape[0] < 3: # Need points for BA
            logging.error("Error: Not enough valid points after initial triangulation for BA.")
            return None

        # Run Bundle Adjustment
        result = self.run_bundle_adjustment(points_2d_observed_filtered, initial_poses)
        if result is None:
            logging.error("Bundle adjustment failed.")
            return None
        optimized_poses, points3D_optimized, final_points_2d = result

        # Update camera poses
        self.set_poses(optimized_poses)
        
        # Compute and log calibration quality metrics
        calibration_quality = self._log_calibration_quality(final_points_2d, points3D_optimized)
        
        logging.info("Calibration completed.")

        return optimized_poses, points3D_optimized, calibration_quality

    def _validate_intrinsics(self):
        """
        Validate loaded intrinsic parameters for reasonableness.
        """
        for i, K in enumerate(self.ks):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Check focal lengths are reasonable (typical range for PS3 Eye: 200-400 pixels)
            if fx < 100 or fx > 800:
                logging.warning(f"Camera {i}: Unusual focal length fx={fx:.1f} (expected 200-400)")
            if fy < 100 or fy > 800:
                logging.warning(f"Camera {i}: Unusual focal length fy={fy:.1f} (expected 200-400)")
            
            # Check principal point is reasonable (should be near image center)
            if cx < 50 or cx > 500:
                logging.warning(f"Camera {i}: Unusual principal point cx={cx:.1f}")
            if cy < 50 or cy > 500:
                logging.warning(f"Camera {i}: Unusual principal point cy={cy:.1f}")
                
            # Check aspect ratio (should be close to 1.0 for square pixels)
            aspect_ratio = fx / fy
            if abs(aspect_ratio - 1.0) > 0.1:
                logging.warning(f"Camera {i}: Non-square pixels, aspect ratio={aspect_ratio:.3f}")
                
            logging.info(f"Camera {i} intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    def _log_calibration_quality(self, points_2d_observed, points_3d_optimized):
        """
        Compute and log calibration quality metrics.
        """
        if points_2d_observed.shape[0] == 0 or points_3d_optimized.shape[0] == 0:
            logging.warning("No points available for quality assessment")
            return
            
        # Calculate reprojection errors for all cameras
        total_errors = []
        camera_errors = {}
        
        for cam_idx in range(self.num_cameras):
            cam_errors = []
            for pt_idx in range(points_3d_optimized.shape[0]):
                if pt_idx < points_2d_observed.shape[0]:
                    observed_2d = points_2d_observed[pt_idx, cam_idx, :]
                    if np.all(np.isfinite(observed_2d)):
                        # Project 3D point to camera
                        projected_2d = self._project(
                            points_3d_optimized[pt_idx:pt_idx+1], 
                            self.ks[cam_idx], 
                            self.Rs[cam_idx], 
                            self.ts[cam_idx], 
                            np.zeros(5)  # Points are in undistorted space
                        )
                        if projected_2d.shape[0] > 0:
                            error = np.linalg.norm(projected_2d[0] - observed_2d)
                            cam_errors.append(error)
                            total_errors.append(error)
            
            if cam_errors:
                camera_errors[cam_idx] = {
                    'mean': np.mean(cam_errors),
                    'std': np.std(cam_errors),
                    'max': np.max(cam_errors),
                    'count': len(cam_errors)
                }

        overall_mean = 0
        overall_std = 0
        overall_max = 0
        
        # Log overall quality metrics
        if total_errors:
            overall_mean = np.mean(total_errors)
            overall_std = np.std(total_errors)
            overall_max = np.max(total_errors)
            
            logging.info(f"=== Calibration Quality Metrics ===")
            logging.info(f"Overall reprojection error: {overall_mean:.2f} ± {overall_std:.2f} pixels (max: {overall_max:.2f})")
            logging.info(f"Total calibration points used: {len(points_3d_optimized)}")
            
            # Quality Assessment
            if overall_mean < 2.0:
                logging.info("✅ EXCELLENT calibration quality (mean < 2px)")
            elif overall_mean < 5.0:
                logging.info("✅ GOOD calibration quality (mean < 5px)")
            elif overall_mean < 10.0:
                logging.warning("⚠️  ACCEPTABLE calibration quality (mean < 10px)")
            else:
                logging.error("❌ POOR calibration quality (mean > 10px) - Recalibration recommended")
            
            for cam_idx, metrics in camera_errors.items():
                logging.info(f"Camera {cam_idx}: {metrics['mean']:.2f} ± {metrics['std']:.2f} pixels "
                           f"(max: {metrics['max']:.2f}, n={metrics['count']})")
            
            # Warn if errors are too high
            if overall_mean > 5.0:
                logging.warning(f"High reprojection errors detected! Consider recalibrating.")
        else:
            logging.warning("Could not compute calibration quality metrics")

        return overall_mean, overall_std, overall_max

    @staticmethod
    def find_drones(points_3d: np.ndarray) -> list[dict]:
        """
        Finds drones in a list of 3D points based on a specific LED configuration.

        A drone is indicated by three LEDs:
        - Two LEDs (P_rear, P_front) are separated by ~15 cm.
        - A third LED (P_left_led) is ~10 cm from the midpoint of P_rear-P_front.
        - The vector from midpoint to P_left_led is perpendicular to P_front-P_rear.
        - P_left_led is "to the left" of the directed line P_rear -> P_front,
          assuming +Y is the world's "up" direction.

        Args:
            points_3d: A NumPy array of 3D points (N, 3).

        Returns:
            A list of dictionaries, where each dictionary represents a found drone
            with 'center' (3D point) and 'direction' (3D unit vector).
        """
        if points_3d is None or points_3d.shape[0] < 3:
            return []

        found_drones_data = []
        # Using a set of frozensets of indices to ensure each unique group of 3 LEDs forms only one drone.
        # This means if points (0,1,2) form a drone, (1,0,2) etc. for the same LEDs won't be added again.
        processed_triplets_indices = set()

        # Constants
        DIST_BASE_PAIR = 0.15  # meters (distance between rear and front LEDs)
        DIST_OFFSET_LED_TO_CENTER = 0.05  # meters (distance from left LED to midpoint of base pair) 
        TOLERANCE = 0.05 
        # Max dot product for perpendicularity (cos(angle_deviation_from_90_deg))
        # e.g., cos(90-10deg) = sin(10deg) = 0.1736. So abs(dot_product) < 0.18 means angle is within 80-100 deg.
        PERPENDICULARITY_DOT_MAX = 0.18
        # Min dot product for "left" alignment (cos(angle_max_deviation_from_expected_left))
        # e.g., cos(30deg) = 0.866.

        num_points = points_3d.shape[0]
        indices = list(range(num_points))

        for p_indices in itertools.permutations(indices, 3):
            idx_rear, idx_front, idx_left_led = p_indices

            # Check if this combination of physical points (LEDs) has already formed a drone
            current_physical_leds_indices = frozenset({idx_rear, idx_front, idx_left_led})
            if current_physical_leds_indices in processed_triplets_indices:
                continue
            
            p_rear = points_3d[idx_rear]
            p_front = points_3d[idx_front]
            p_left = points_3d[idx_left_led]

            # 1. Check distance between P_rear and P_front
            vec_base = p_front - p_rear
            norm_vec_base = np.linalg.norm(vec_base)
            if abs(norm_vec_base - DIST_BASE_PAIR) > TOLERANCE:
                continue
            
            # 2. Calculate midpoint and check distance from P_left_led to midpoint
            midpoint_base = (p_rear + p_front) / 2.0
            vec_offset_actual = p_left - midpoint_base
            norm_vec_offset_actual = np.linalg.norm(vec_offset_actual)
            if abs(norm_vec_offset_actual - DIST_OFFSET_LED_TO_CENTER) > TOLERANCE:
                continue
            

            # 3. Check perpendicularity of offset vector to base vector
            if norm_vec_base < 1e-6 or norm_vec_offset_actual < 1e-6: # Avoid division by zero
                continue

            dir_base_norm = vec_base / norm_vec_base
            dir_offset_norm = vec_offset_actual / norm_vec_offset_actual

            # Ensure dir_base_norm and dir_offset_norm do not contain NaNs after division
            if np.isnan(dir_base_norm).any() or np.isnan(dir_offset_norm).any():
                continue

            cos_angle_perp = np.dot(dir_base_norm, dir_offset_norm)
            if abs(cos_angle_perp) > PERPENDICULARITY_DOT_MAX: # Check if vectors are nearly perpendicular
                continue
            
            # If all geometric checks pass, this triplet forms a drone.
            # The "left" LED is implicitly p_left, and the direction is from p_rear to p_front.
            drone_center = midpoint_base
            drone_direction = dir_base_norm # Normalized direction from P_rear to P_front

            # 4. Check if the direction is "left" of the base pair
            mid = (p_rear + p_front) / 2.0
            vec_base = p_front - p_rear
            vec_offset = p_left - mid
            cross_y = np.dot(np.cross(vec_base, vec_offset), np.array([0, 1, 0]))
            if cross_y < 0:
                drone_direction = -drone_direction

            found_drones_data.append({
                'center': drone_center, # Convert to list for JSON serialization
                'direction': drone_direction # Convert to list for JSON serialization
            })
            processed_triplets_indices.add(current_physical_leds_indices)
        
        return found_drones_data


    @staticmethod
    def _project(points_3d, K, R, t, dist_coeffs) -> np.ndarray:
        """
        Project 3D points into a camera view.

        Args:
            points_3d: numpy array of 3D points. (n_points, 3)
            K: camera intrinsic matrix.
            R: rotation matrix.
            t: translation vector.
            dist_coeffs: distortion coefficients.

        Returns:
            numpy array of 2D points. (n_points, 2)
        """
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
    

    @staticmethod
    def _triangulate_point_dlt(points_2d_all_views, Rs, ts, Ks) -> np.ndarray:
        """
        Triangulates a single 3D point from multiple 2D observations using DLT.

        Args:
            points_2d_all_views: numpy array of 2D points. (n_views, 2)
            Rs: list of rotation matrices.
            ts: list of translation vectors.
            Ks: list of camera intrinsic matrices.

        Returns:
            numpy array of 3D point. (3,)
        """
        n_views = len(points_2d_all_views)
        if n_views != len(Rs) or n_views != len(ts) or n_views != len(Ks):
            raise ValueError("Number of points, poses, and intrinsics must match.")

        # Filter out invalid points (NaN/Inf) and corresponding poses/Ks
        valid_mask = np.all(np.isfinite(points_2d_all_views), axis=1)
        if np.sum(valid_mask) < 2:
            return np.array([np.nan, np.nan, np.nan]) # Need at least 2 views

        points_2d_valid = points_2d_all_views[valid_mask]
        Rs_valid = [Rs[i] for i, valid in enumerate(valid_mask) if valid]
        ts_valid = [ts[i] for i, valid in enumerate(valid_mask) if valid]
        Ks_valid = [Ks[i] for i, valid in enumerate(valid_mask) if valid]

        # Construct projection matrices
        Ps = []
        for i, (K, R, t) in enumerate(zip(Ks_valid, Rs_valid, ts_valid)):
            P = K @ np.hstack((R, t))
            Ps.append(P)
            # logging.info(f"Camera {i} - R shape: {R.shape}, t shape: {t.shape}")
            if np.any(np.isnan(R)) or np.any(np.isnan(t)):
                logging.warning(f"Camera {i} has NaN values in R or t!")
            
        # logging.info(f"Triangulating with {len(Ps)} cameras, {len(points_2d_valid)} 2D points")

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
        if A.shape[0] < 4:
            logging.warning(f"Warning: A matrix too small for triangulation: {A.shape}")
            return np.array([np.nan, np.nan, np.nan])
            
        U, s, Vh = linalg.svd(A)
        # The 3D point in homogeneous coordinates is the last row of Vh (or last col of V)
        point_4d_hom = Vh[-1]
        
        # logging.info(f"SVD results - singular values: {s}")
        # logging.info(f"Homogeneous solution: {point_4d_hom}")

        # Dehomogenize - use more relaxed threshold
        if abs(point_4d_hom[3]) < 1e-8: # More relaxed threshold
            logging.warning(f"Warning: Near-zero homogeneous coordinate: {point_4d_hom[3]}")
            return np.array([np.nan, np.nan, np.nan])

        point_3d = point_4d_hom[:3] / point_4d_hom[3]
        
        # Re-enable validation with more reasonable threshold for initial estimates
        # Use lenient threshold since bundle adjustment will refine these estimates
        if not Cameras._validate_triangulated_point(point_3d, points_2d_all_views, Rs, ts, Ks, max_reprojection_error=100.0):
            return np.array([np.nan, np.nan, np.nan])
        
        return point_3d


    @staticmethod
    def _triangulate_points(points_2d_multi_cam, Rs, ts, Ks) -> np.ndarray:
        """
        Triangulates multiple 3D points from N camera views.

        Args:
            points_2d_multi_cam: numpy array of 2D points. (n_points, n_cameras, 2)
            Rs: list of rotation matrices.
            ts: list of translation vectors.
            Ks: list of camera intrinsic matrices.

        Returns:
            numpy array of 3D points. (n_points, 3)
        """
        n_points = points_2d_multi_cam.shape[0]
        if n_points == 0:
            return np.empty((0, 3))

        points_3d = np.zeros((n_points, 3))
        for i in range(n_points):
            # points_2d_multi_cam[i] has shape (n_cameras, 2)
            points_3d[i] = Cameras._triangulate_point_dlt(points_2d_multi_cam[i], Rs, ts, Ks)

        return points_3d
    

    @staticmethod
    def _validate_triangulated_point(point_3d, points_2d_observed, Rs, ts, Ks, max_reprojection_error=10.0) -> bool:
        """
        Validate a triangulated 3D point by checking reprojection errors.
        
        Args:
            point_3d: The triangulated 3D point (3,)
            points_2d_observed: Original 2D observations (n_views, 2)
            Rs, ts, Ks: Camera parameters
            max_reprojection_error: Maximum allowed reprojection error in pixels
            
        Returns:
            True if point is valid, False otherwise
        """
        if not np.all(np.isfinite(point_3d)):
            return False
            
        # Check if point is at reasonable distance (not too close or too far)
        point_distance = np.linalg.norm(point_3d)
        if point_distance < 0.1 or point_distance > 100.0:  # 10cm to 100m reasonable range
            return False
        
        # Calculate reprojection errors for all valid views
        reprojection_errors = []
        valid_mask = np.all(np.isfinite(points_2d_observed), axis=1)
        
        for i, (R, t, K) in enumerate(zip(Rs, ts, Ks)):
            if not valid_mask[i]:
                continue
                
            # Project 3D point to this camera
            projected_2d = Cameras._project(point_3d.reshape(1, 3), K, R, t, np.zeros(5))
            if projected_2d.shape[0] > 0:
                error = np.linalg.norm(projected_2d[0] - points_2d_observed[i])
                reprojection_errors.append(error)
        
        if not reprojection_errors:
            return False
            
        # Check if median reprojection error is reasonable
        median_error = np.median(reprojection_errors)
        return median_error < max_reprojection_error

    @staticmethod
    def _preprocess_frame(frame, rotation, k, distortion_coefficients) -> np.ndarray:
        """
        Preprocesses a frame for triangulation.

        Args:
            frame: numpy array of image. (H, W, 3)
            rotation: rotation angle.
            k: camera intrinsic matrix.
            distortion_coefficients: camera distortion coefficients.

        Returns:
            numpy array of image. (H, W, 3)
        """
        frame = np.rot90(frame, k=rotation)
        frame = Cameras._make_square(frame)
        frame = cv2.undistort(frame, k, distortion_coefficients)
        frame = cv2.GaussianBlur(frame,(9,9),0)
        frame = cv2.filter2D(frame, -1, Cameras.KERNEL)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    @staticmethod
    def _find_dots(img) -> tuple[np.ndarray, np.ndarray]:
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
                center_x = moments["m10"] / moments["m00"]
                center_y = moments["m01"] / moments["m00"]
                criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
                # Convert grey to float32 for cornerSubPix if it's not already
                grey_float = grey.astype(np.float32) if grey.dtype != np.float32 else grey
                subpix = cv2.cornerSubPix(grey_float, np.array([[center_x,center_y]],np.float32), (5,5), (-1,-1), criteria)
                image_points.append([subpix[0][0], subpix[0][1]])

                # For visualization, convert subpixel coordinates to int
                viz_center_x = int(round(subpix[0][0]))
                viz_center_y = int(round(subpix[0][1]))
                cv2.putText(img, f'{subpix[0][0]:.2f}, {subpix[0][1]:.2f}', (viz_center_x, viz_center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                cv2.circle(img, (viz_center_x, viz_center_y), 1, (100,255,100), -1)


        if len(image_points) == 0:
            image_points = [[np.nan, np.nan]] # Use np.nan for missing float points

        return img, np.array(image_points, dtype=np.float32)
    
    @staticmethod
    def _make_square(img) -> np.ndarray:
        """
        Makes a square image. If the image is not square, it will be padded with edge pixels. 
        Then, it will be feathered to avoid visible edges.

        Args:
            img: numpy array of image. (H, W, 3)

        Returns:
            numpy array of image. (H, W, 3)
        """
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
    

if __name__ == "__main__":
    cameras = Cameras(num_cameras=4)
    cameras.calibrate_cameras()
