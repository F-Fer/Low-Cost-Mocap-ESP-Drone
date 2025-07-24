#!/usr/bin/env python3
"""
Individual Camera Calibration Script for PS3 Eye Cameras

This script calibrates each of the 4 PS3 Eye cameras individually using a 7x10 chessboard pattern.
It captures multiple images of the chessboard from different angles and positions, then calculates
the intrinsic parameters (camera matrix and distortion coefficients) for each camera.

The calibration results are saved to camera-params.json in the format expected by the main system.
"""

import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from pseyepy import Camera
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndividualCameraCalibrator:
    """Calibrates PS3 Eye cameras individually using chessboard patterns."""
    
    # Camera settings (matching cameras.py)
    FPS = 60
    # EXPOSURE = 250
    # GAIN = 30
    RESOLUTION = Camera.RES_SMALL  # 320x240
    
    # Chessboard settings
    CHESSBOARD_SIZE = (6, 9)  # Internal corners (width x height)
    SQUARE_SIZE = 1.0  # Size of chessboard squares in arbitrary units
    
    # Calibration settings
    MIN_IMAGES = 15  # Minimum number of good images for calibration
    MAX_IMAGES = 30  # Maximum number of images to capture
    
    def __init__(self, camera_id, output_dir="./"):
        """
        Initialize the calibrator for a single camera.
        
        Args:
            camera_id: ID of the camera to calibrate (0-3)
            output_dir: Directory to save calibration results
        """
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / f"camera_{camera_id}_params.json"
        self.camera = None
        
        # Prepare object points for chessboard
        self.objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp *= self.SQUARE_SIZE
        
        # Storage for calibration data
        self.camera_params = None
        
    def initialize_camera(self):
        """Initialize the PS3 Eye camera."""
        try:
            self.camera = Camera(
                fps=self.FPS,
                resolution=self.RESOLUTION,
                # gain=self.GAIN,
                # exposure=self.EXPOSURE,
                colour=True
            )
            logging.info("Camera initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_calibration_images(self):
        """
        Capture calibration images for the specified camera.
            
        Returns:
            Tuple of (object_points, image_points, image_size)
        """
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        image_size = None
        camera_id = self.camera_id
        
        print(f"\n=== Calibrating Camera {camera_id} ===")
        print(f"Target: {self.MIN_IMAGES}-{self.MAX_IMAGES} good images")
        print("Instructions:")
        print("- Hold the chessboard in front of the camera")
        print("- Move it to different positions and angles")
        print("- Press SPACE to capture when chessboard is detected")
        print("- Press 'q' to finish calibration for this camera")
        print("- Press 'r' to retry current capture")
        
        captured_count = 0
        
        while captured_count < self.MAX_IMAGES:
            # Read frame from camera
            frames, _ = self.camera.read()
            if frames is None or len(frames) <= camera_id:
                logging.warning("No frame received from camera")
                time.sleep(0.1)
                continue
                
            frame = frames#[camera_id]
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])  # (width, height)

            # Convert to grayscale for chessboard detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.CHESSBOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Create display frame
            display_frame = frame.copy()
            
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw chessboard corners
                cv2.drawChessboardCorners(display_frame, self.CHESSBOARD_SIZE, corners_refined, ret)
                
                # Add status text
                status_text = f"Camera {camera_id} - READY TO CAPTURE ({captured_count}/{self.MAX_IMAGES})"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Add status text for no detection
                status_text = f"Camera {camera_id} - NO CHESSBOARD DETECTED ({captured_count}/{self.MAX_IMAGES})"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Position chessboard in view", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Show progress
            cv2.putText(display_frame, f"Min required: {self.MIN_IMAGES}", (10, display_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'q' to finish", (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow(f'Camera {camera_id} Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret:  # Space key - capture image
                objpoints.append(self.objp)
                imgpoints.append(corners_refined)
                captured_count += 1
                
                logging.info(f"Camera {camera_id}: Captured image {captured_count}/{self.MAX_IMAGES}")
                
                # Brief feedback
                feedback_frame = display_frame.copy()
                cv2.putText(feedback_frame, "CAPTURED!", (display_frame.shape[1]//2 - 80, display_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow(f'Camera {camera_id} Calibration', feedback_frame)
                cv2.waitKey(500)  # Show feedback for 500ms
                
            elif key == ord('q'):  # Quit calibration for this camera
                if captured_count >= self.MIN_IMAGES:
                    break
                else:
                    print(f"Need at least {self.MIN_IMAGES} images. Currently have {captured_count}.")
                    print("Press 'q' again to force quit, or continue capturing.")
                    if cv2.waitKey(2000) & 0xFF == ord('q'):  # Wait 2 seconds for second 'q'
                        break
            
            elif key == ord('r') and ret:  # Retry - remove last captured image
                if captured_count > 0:
                    objpoints.pop()
                    imgpoints.pop()
                    captured_count -= 1
                    logging.info(f"Camera {camera_id}: Removed last capture. Now have {captured_count} images.")
            
            time.sleep(1/self.FPS)  # Control frame rate
        
        cv2.destroyAllWindows()
        
        if captured_count < self.MIN_IMAGES:
            logging.warning(f"Camera {camera_id}: Only captured {captured_count} images (minimum {self.MIN_IMAGES})")
            return None, None, None
        
        logging.info(f"Camera {camera_id}: Calibration capture complete with {captured_count} images")
        return objpoints, imgpoints, image_size
    
    def calibrate_single_camera(self, objpoints, imgpoints, image_size):
        """
        Perform calibration for a single camera.
        
        Args:
            objpoints: List of object points
            imgpoints: List of image points
            image_size: Size of the images (width, height)
            
        Returns:
            Dictionary with calibration parameters
        """
        camera_id = self.camera_id
        logging.info(f"Running calibration for Camera {camera_id}...")
        
        # Initial camera matrix guess
        initial_camera_matrix = np.array([
            [image_size[0], 0, image_size[0]/2],
            [0, image_size[0], image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Calibration flags
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
                cv2.CALIB_RATIONAL_MODEL +
                cv2.CALIB_FIX_ASPECT_RATIO)
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, initial_camera_matrix, None, flags=flags
        )
        
        if not ret:
            logging.error(f"Camera {camera_id}: Calibration failed")
            return None
        
        # Calculate reprojection error
        total_error = 0
        total_points = 0
        
        for i in range(len(objpoints)):
            imgpoints_proj, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            total_error += error
            total_points += 1
        
        mean_error = total_error / total_points
        logging.info(f"Camera {camera_id}: Mean reprojection error: {mean_error:.3f} pixels")
        
        # Store calibration results
        calibration_data = {
            "camera_id": camera_id,
            "intrinsic_matrix": camera_matrix.tolist(),
            "distortion_coef": dist_coeffs.flatten().tolist(),
            "rotation": 0,  # Default rotation, can be adjusted later
            "reprojection_error": mean_error,
            "num_images": len(objpoints),
            "image_size": list(image_size),
            "calibration_date": datetime.now().isoformat()
        }
        
        return calibration_data
    
    def calibrate_camera(self):
        """Calibrate the specified camera."""
        if not self.initialize_camera():
            return False
        
        try:
            camera_id = self.camera_id
            print(f"\n{'='*50}")
            print(f"Starting calibration for Camera {camera_id}")
            print(f"{'='*50}")
            
            # Capture calibration images
            objpoints, imgpoints, image_size = self.capture_calibration_images()
            
            if objpoints is None:
                print(f"Camera {camera_id} calibration failed - not enough images captured.")
                return False
            
            # Perform calibration
            calibration_data = self.calibrate_single_camera(objpoints, imgpoints, image_size)
            
            if calibration_data is None:
                print(f"Camera {camera_id} calibration computation failed.")
                return False
            
            # Store results
            self.camera_params = calibration_data
            
            print(f"Camera {camera_id} calibration completed successfully!")
            print(f"Reprojection error: {calibration_data['reprojection_error']:.3f} pixels")
            return True
        
        finally:
            if self.camera:
                # pseyepy Camera objects don't have a close() method
                # The camera will be cleaned up automatically
                pass
    
    def save_calibration_results(self):
        """Save calibration results to JSON file."""
        if not self.camera_params:
            logging.error("No calibration data to save")
            return False
        
        # Create backup if file exists
        output_path = Path(self.output_file)
        if output_path.exists():
            backup_path = output_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            output_path.rename(backup_path)
            logging.info(f"Existing calibration file backed up to: {backup_path}")
        
        # Save new calibration data
        try:
            with open(output_path, 'w') as f:
                json.dump(self.camera_params, f, indent=2)
            
            logging.info(f"Calibration results saved to: {output_path}")
            
            # Print summary
            print(f"\n{'='*50}")
            print("CALIBRATION SUMMARY")
            print(f"{'='*50}")
            
            camera_id = self.camera_id
            data = self.camera_params
            print(f"Camera {camera_id}:")
            print(f"  - Images used: {data['num_images']}")
            print(f"  - Reprojection error: {data['reprojection_error']:.3f} pixels")
            print(f"  - Image size: {data['image_size']}")
            
            print(f"\nCalibration data saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save calibration results: {e}")
            return False
    
    def load_and_display_results(self):
        """Load and display existing calibration results."""
        try:
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            
            print(f"\n{'='*50}")
            print(f"EXISTING CALIBRATION DATA - Camera {self.camera_id}")
            print(f"{'='*50}")
            
            if isinstance(data, dict):
                print(f"  - Reprojection error: {data.get('reprojection_error', 'N/A')}")
                print(f"  - Images used: {data.get('num_images', 'N/A')}")
                print(f"  - Calibration date: {data.get('calibration_date', 'N/A')}")
                print(f"  - Image size: {data.get('image_size', 'N/A')}")
            else:
                print(f"  - Legacy format detected")
            print()
                
        except FileNotFoundError:
            print(f"No existing calibration file found at: {self.output_file}")
        except Exception as e:
            print(f"Error reading calibration file: {e}")


def get_camera_id():
    """Get camera ID from user input."""
    while True:
        try:
            camera_id = int(input("Enter camera ID to calibrate (0-3): "))
            if 0 <= camera_id <= 3:
                return camera_id
            else:
                print("Camera ID must be between 0 and 3.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    """Main function to run the calibration process."""
    print("PS3 Eye Camera Individual Calibration Script")
    print("=" * 50)
    
    # Get camera ID from user
    camera_id = get_camera_id()
    
    calibrator = IndividualCameraCalibrator(camera_id)
    
    # Check for existing calibration
    if Path(calibrator.output_file).exists():
        print(f"Existing calibration file found for Camera {camera_id}!")
        calibrator.load_and_display_results()
        
        response = input(f"\nOverwrite existing calibration for Camera {camera_id}? (y/n): ")
        if response.lower() != 'y':
            print("Calibration cancelled.")
            return
    
    # Run calibration
    print(f"\nStarting calibration process for Camera {camera_id}...")
    print("Make sure your 7x10 chessboard is ready!")
    input("Press Enter to begin...")
    
    success = calibrator.calibrate_camera()
    
    if success:
        calibrator.save_calibration_results()
        print(f"\nCalibration process completed for Camera {camera_id}!")
        print("\nTo calibrate other cameras, run this script again with different camera IDs.")
        print("Once all cameras are calibrated, you can combine the results into camera-params.json.")
    else:
        print(f"\nCalibration process failed for Camera {camera_id}.")


if __name__ == "__main__":
    main()
