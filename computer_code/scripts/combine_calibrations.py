#!/usr/bin/env python3
"""
Combine Individual Camera Calibrations

This script combines individual camera calibration files (camera_0_params.json, camera_1_params.json, etc.)
into the main camera-params.json file expected by the motion capture system.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_calibrations(input_dir="./", output_file="camera-params.json", num_cameras=4):
    """
    Combine individual camera calibration files into a single file.
    
    Args:
        input_dir: Directory containing individual calibration files
        output_file: Output file path for combined calibrations
        num_cameras: Number of cameras to combine (0 to num_cameras-1)
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    combined_params = [None] * num_cameras  # Initialize list with None values
    found_cameras = []
    
    # Load individual camera calibration files
    for camera_id in range(num_cameras):
        camera_file = input_path / f"camera_{camera_id}_params.json"
        
        if camera_file.exists():
            try:
                with open(camera_file, 'r') as f:
                    camera_data = json.load(f)
                
                # Store in list at the correct index
                combined_params[camera_id] = camera_data
                found_cameras.append(camera_id)
                logging.info(f"Loaded calibration for Camera {camera_id}")
                
            except Exception as e:
                logging.error(f"Failed to load calibration for Camera {camera_id}: {e}")
        else:
            logging.warning(f"Calibration file not found for Camera {camera_id}: {camera_file}")
    
    if not found_cameras:
        logging.error("No camera calibration files found!")
        return False
    
    # Remove None values for cameras that weren't found, keeping only found cameras
    combined_params = [param for param in combined_params if param is not None]
    
    # Create backup if output file exists
    if output_path.exists():
        backup_path = output_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        output_path.rename(backup_path)
        logging.info(f"Existing file backed up to: {backup_path}")
    
    # Save combined calibration data
    try:
        with open(output_path, 'w') as f:
            json.dump(combined_params, f, indent=2)
        
        logging.info(f"Combined calibration saved to: {output_path}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("COMBINED CALIBRATION SUMMARY")
        print(f"{'='*50}")
        print(f"Successfully combined {len(found_cameras)} camera calibrations:")
        
        for i, data in enumerate(combined_params):
            camera_id = data.get('camera_id', i)
            print(f"Camera {camera_id}:")
            print(f"  - Images used: {data.get('num_images', 'N/A')}")
            print(f"  - Reprojection error: {data.get('reprojection_error', 'N/A'):.3f} pixels")
            print(f"  - Image size: {data.get('image_size', 'N/A')}")
            print(f"  - Calibration date: {data.get('calibration_date', 'N/A')}")
            print()
        
        if len(found_cameras) < num_cameras:
            print(f"WARNING: Only {len(found_cameras)} out of {num_cameras} cameras calibrated.")
            missing = [i for i in range(num_cameras) if i not in found_cameras]
            print(f"Missing calibrations for cameras: {missing}")
        
        print(f"Combined calibration data saved to: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save combined calibration: {e}")
        return False

def main():
    """Main function to combine calibrations."""
    print("Camera Calibration Combiner")
    print("=" * 50)
    
    # Check for individual calibration files
    current_dir = Path(".")
    found_files = []
    
    for camera_id in range(4):
        camera_file = current_dir / f"camera_{camera_id}_params.json"
        if camera_file.exists():
            found_files.append(camera_id)
    
    if not found_files:
        print("No individual camera calibration files found!")
        print("Expected files: camera_0_params.json, camera_1_params.json, etc.")
        return
    
    print(f"Found calibration files for cameras: {found_files}")
    
    # Check if main calibration file exists
    main_file = Path("camera-params.json")
    if main_file.exists():
        print("Existing camera-params.json found!")
        response = input("Overwrite existing camera-params.json? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Combine calibrations
    success = combine_calibrations()
    
    if success:
        print("\nCalibration combination completed successfully!")
        print("You can now use the combined camera-params.json with your motion capture system.")
    else:
        print("\nCalibration combination failed.")

if __name__ == "__main__":
    main() 