import threading
import sys
from flask import Flask, request
from flask_socketio import SocketIO
import base64
import cv2
from flask_cors import CORS  # Import CORS
from pathlib import Path
import numpy as np
from cameras import Cameras
from flask import jsonify
from datetime import datetime
import logging
import time

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).parent)
sys.path.insert(0, parent_dir)

# Calibration points backlog
calibration_points_ndarray = np.load("calibration_points.npy") # (num_points, num_cameras, 2)
calibration_points = []
for i in range(calibration_points_ndarray.shape[0]):
    calibration_points.append(calibration_points_ndarray[i, :, :].tolist())

# Stream calibration state
stream_calibration_active = False
stream_calibration_points = []
stream_calibration_stats = {
    'total_frames_processed': 0,
    'valid_frames_collected': 0,
    'collection_rate': 0.0,
    'start_time': None
}

camera_params_dir = "camera-params.json"
# Find the latest poses file
poses_dir = "poses"
pose_files = sorted(Path(poses_dir).glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
if not pose_files:
    raise FileNotFoundError("No pose files found in the 'poses' directory.")
camera_poses_dir = str(pose_files[0])
logging.info(f"Taking poses from {camera_poses_dir}")

num_cameras = 4
cameras = Cameras(num_cameras=num_cameras, camera_params_path=camera_params_dir)
cameras.load_poses(camera_poses_dir)

# Get the poses for each camera
Rs, ts = cameras.Rs, cameras.ts
camera_data = []
for i, (R, t) in enumerate(zip(Rs, ts)):
    camera_data.append({"id": f"c{i}", "position": t.tolist(), "rotation": R.tolist()})

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!' # Change this!

# Configure CORS to allow requests from your frontend origin
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://172.20.10.4:3000", "http://141.75.222.154:3000"])

# Initialize SocketIO, allowing async_mode and CORS
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://172.20.10.4:3000", "http://141.75.222.154:3000"], async_mode='eventlet')

points_data = [
    {"id": "p1", "position": [0, 0, 0]},
    {"id": "p2", "position": [1, 1, 1]},
    {"id": "p3", "position": [-1, 0, 1]},
]
data_lock = threading.Lock() # Lock for thread-safe access to points_data
stop_event = threading.Event()

def stream_cameras_and_points():
    """Stream filtered drone tracking data."""
    logging.info("Starting filtered drone tracking thread...")
    
    # Use the new filtered drone streaming
    filtered_drones_iterator = cameras.stream_filtered_drones()
    
    while not stop_event.is_set():
        try:
            # Get filtered drone states
            tracked_drones, points_3d = next(filtered_drones_iterator)
            
            # Convert tracked drones to the expected format
            drones_data = []
            for drone_state in tracked_drones:
                drones_data.append({
                    "id": f"track_{drone_state['track_id']}",
                    "position": drone_state['position'].tolist(),
                    "velocity": drone_state['velocity'].tolist(),
                    "heading": drone_state['heading'],
                    "confidence": drone_state['confidence'],
                    "track_id": drone_state['track_id'],
                    "age": drone_state['age']
                })
            
            # Also get raw points for visualization
            points_data = []
            for i, point in enumerate(points_3d):
                if np.all(np.isfinite(point)):
                    points_data.append({"id": f"p{i}", "position": point.tolist()})

            # Convert tracked drones to old format for backward compatibility
            old_format_drones = []
            for drone_state in tracked_drones:
                old_format_drones.append({
                    "id": f"d{drone_state['track_id']}",
                    "position": drone_state['position'].tolist(),
                    "direction": [np.cos(drone_state['heading']), np.sin(drone_state['heading']), 0]
                })

            # Emit the updated data to all connected clients
            socketio.emit('update_points', {
                'points': points_data, 
                'cameras': camera_data, 
                'drones': old_format_drones,  # Old format for compatibility
                'tracked_drones': drones_data,  # New format with full tracking info
                'tracking_stats': cameras.get_tracking_statistics()
            })
            
        except Exception as e:
            logging.error(f"Error in stream_cameras_and_points: {e}")
            
        socketio.sleep(1/cameras.FPS)
    logging.info("Point update thread stopped.")


def stream_camera_feeds():
    """Stream camera frames via Socket.IO and collect calibration points when stream is active."""
    global stream_calibration_active, stream_calibration_points, stream_calibration_stats
    
    logging.info("Starting camera feed streaming thread...")
    
    while not stop_event.is_set():
        try:
            # Always get fresh frames for display
            display_frames, _ = cameras.cameras.read()
            
            # Handle stream calibration if active (separate from display frames)
            if stream_calibration_active:
                # Use the simplified single-point collection method for stream calibration
                cal_frames, single_points = cameras.collect_single_point_frame()
                
                # Process every frame attempt for statistics
                stream_calibration_stats['total_frames_processed'] += 1
                
                if single_points is not None:
                    # Successfully collected points from all cameras - add to collection
                    stream_calibration_points.append(single_points)
                    stream_calibration_stats['valid_frames_collected'] += 1
                    
                    # Emit calibration status update
                    duration = time.time() - stream_calibration_stats['start_time']
                    current_rate = len(stream_calibration_points) / duration if duration > 0 else 0
                    
                    socketio.emit('calibration_stream_update', {
                        'points_collected': len(stream_calibration_points),
                        'total_frames_processed': stream_calibration_stats['total_frames_processed'],
                        'collection_rate': current_rate,
                        'duration': duration,
                        'valid_frame': True
                    })
                else:
                    # Frame didn't have exactly one point per camera - emit update
                    socketio.emit('calibration_stream_update', {
                        'points_collected': len(stream_calibration_points),
                        'total_frames_processed': stream_calibration_stats['total_frames_processed'],
                        'valid_frame': False
                    })
            
            # Use display frames for camera feed streaming
            frames = display_frames
            
            # Process and emit each frame for display
            if frames is not None and len(frames) > 0:
                for i, frame in enumerate(frames):
                    if frame is not None and i < cameras.num_cameras:
                        try:
                            # Preprocess frame
                            frame = Cameras._preprocess_frame(frame, cameras.rotations[i], cameras.ks[i], cameras.distortion_coefficients[i])
                            frame, _ = Cameras._find_dots(frame)
                            
                            # Convert frame to JPEG format
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            
                            # Convert to base64 for sending via Socket.IO
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Emit the frame
                            socketio.emit('camera_frame', {
                                'camera_id': i,
                                'frame': frame_b64
                            })
                        except Exception as frame_error:
                            logging.debug(f"Error processing frame {i}: {frame_error}")
                            continue
            
            # Control frame rate
            socketio.sleep(1/30)  # Limit to 30fps to reduce bandwidth
        except Exception as e:
            logging.error(f"Error streaming camera feeds: {e}")
            socketio.sleep(1)  # Wait before retrying
    
    logging.info("Camera feed streaming thread stopped.")


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/capture_calibration_points', methods=['POST'])
def capture_calibration_points():
    _, points = cameras.collect_calibration_points()
    if points is None:
        return jsonify({"error": "Not all cameras detected a point."}), 400
    calibration_points.append(points)
    return jsonify({"message": "Calibration points captured."}), 200


@app.route('/get_num_calibration_points', methods=['GET'])
def get_num_calibration_points():
    return jsonify({"num_calibration_points": len(calibration_points), "min_calibration_points": cameras.MIN_CALIBRATION_POINTS}), 200


@app.route('/start_calibration', methods=['POST'])
def start_calibration():
    global camera_data
    np.save("calibration_points.npy", np.array(calibration_points))

    if len(calibration_points) < cameras.MIN_CALIBRATION_POINTS:
        return jsonify({"error": "Not enough calibration points collected."}), 400
    
    points_array = np.array(calibration_points)
    optimized_poses, points3D_optimized, calibration_quality = cameras.calibrate_cameras(points_array)
    mean, std, max = calibration_quality
    if optimized_poses is None:
        return jsonify({"error": "Calibration failed. See server logs for details."}), 500
    # Update camera_data with the optimized poses
    Rs, ts = cameras.Rs, cameras.ts
    camera_data = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        camera_data.append({"id": f"c{i}", "position": t.tolist(), "rotation": R.tolist()})
    # Save the optimized poses to a json file
    cameras.save_poses()
    message = f"Calibration completed. (mean: {mean:.2f} ± {std:.2f} pixels, max: {max:.2f})"
    print("message", message)
    return jsonify({"message": message}), 200


@app.route('/clear_calibration_points', methods=['POST'])
def clear_calibration_points():
    global calibration_points
    calibration_points = []
    return jsonify({"message": "Calibration points cleared."}), 200


@app.route('/set_scale_factor', methods=['POST'])
def set_scale_factor():
    global camera_data
    result = cameras.try_scale_poses()
    if not result:
        return jsonify({"error": "Failed to scale poses."}), 500
    
    cameras.save_poses()

    # Update camera_data with the optimized poses
    Rs, ts = cameras.Rs, cameras.ts
    camera_data = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        camera_data.append({"id": f"c{i}", "position": t.tolist(), "rotation": R.tolist()})
    return jsonify({"message": "Scale factor set."}), 200


@app.route('/set_ground_height', methods=['POST'])
def set_ground_height():
    global camera_data
    ground_height = request.json.get('ground_height')
    if ground_height is None:
        return jsonify({"error": "Ground height not provided."}), 400
    cameras.set_ground(ground_height)
    Rs, ts = cameras.Rs, cameras.ts
    camera_data = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        camera_data.append({"id": f"c{i}", "position": t.tolist(), "rotation": R.tolist()})
    return jsonify({"message": "Ground height set."}), 200


@app.route('/start_calibration_stream', methods=['POST'])
def start_calibration_stream():
    global stream_calibration_active, stream_calibration_points, stream_calibration_stats
    
    if stream_calibration_active:
        return jsonify({"error": "Calibration stream already active."}), 400
    
    # Initialize stream calibration
    stream_calibration_active = True
    stream_calibration_points = []
    stream_calibration_stats = {
        'total_frames_processed': 0,
        'valid_frames_collected': 0,
        'collection_rate': 0.0,
        'start_time': time.time()
    }
    
    logging.info("Started calibration stream collection")
    return jsonify({"message": "Calibration stream started."}), 200


@app.route('/stop_calibration_stream', methods=['POST'])
def stop_calibration_stream():
    global stream_calibration_active, stream_calibration_points, calibration_points
    
    if not stream_calibration_active:
        return jsonify({"error": "No calibration stream active."}), 400
    
    # Stop stream and process collected points
    stream_calibration_active = False
    collected_count = len(stream_calibration_points)
    
    try:
        # Add collected single-point frames directly to calibration points
        if stream_calibration_points:
            # stream_calibration_points is now a list of (num_cameras, 2) arrays
            # Convert to the expected (num_points, num_cameras, 2) format
            calibration_points.extend(stream_calibration_points)
            logging.info(f"Added {collected_count} single-point frames directly to calibration points")
        
        # Calculate final statistics
        duration = time.time() - stream_calibration_stats['start_time']
        final_rate = collected_count / duration if duration > 0 else 0
        
        logging.info(f"Stopped calibration stream. Collected {collected_count} single-point frames in {duration:.1f}s (rate: {final_rate:.1f} frames/s)")
        
        return jsonify({
            "message": "Calibration stream stopped.",
            "single_point_frames_collected": collected_count,
            "calibration_points_added": collected_count,
            "total_points": len(calibration_points),
            "collection_rate": final_rate,
            "duration": duration
        }), 200
        
    except Exception as e:
        logging.error(f"Error processing stream calibration data: {e}")
        return jsonify({
            "error": f"Error processing calibration data: {str(e)}",
            "single_point_frames_collected": collected_count,
            "duration": time.time() - stream_calibration_stats['start_time']
        }), 500


@app.route('/get_calibration_stream_status', methods=['GET'])
def get_calibration_stream_status():
    global stream_calibration_active, stream_calibration_points, stream_calibration_stats
    
    current_time = time.time()
    duration = current_time - stream_calibration_stats['start_time'] if stream_calibration_stats['start_time'] else 0
    current_rate = len(stream_calibration_points) / duration if duration > 0 else 0
    
    return jsonify({
        "active": stream_calibration_active,
        "points_collected": len(stream_calibration_points),
        "total_frames_processed": stream_calibration_stats['total_frames_processed'],
        "collection_rate": current_rate,
        "duration": duration
    }), 200


@app.route('/reset_drone_tracker', methods=['POST'])
def reset_drone_tracker():
    """Reset the drone tracking system."""
    cameras.reset_drone_tracker()
    return jsonify({"message": "Drone tracker reset successfully."}), 200


@app.route('/get_tracking_stats', methods=['GET'])
def get_tracking_stats():
    """Get current tracking statistics."""
    stats = cameras.get_tracking_statistics()
    return jsonify(stats), 200


@app.route('/configure_tracking', methods=['POST'])
def configure_tracking():
    """Configure tracking parameters."""
    data = request.json
    
    max_distance = data.get('max_association_distance')
    timeout = data.get('track_timeout') 
    cutoff_freq = data.get('cutoff_frequency')
    
    try:
        cameras.configure_tracking(
            max_association_distance=max_distance,
            track_timeout=timeout,
            cutoff_frequency=cutoff_freq
        )
        
        return jsonify({
            "message": "Tracking configuration updated successfully.",
            "applied_config": {
                "max_association_distance": max_distance,
                "track_timeout": timeout,
                "cutoff_frequency": cutoff_freq
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to update tracking configuration: {str(e)}"}), 500


@socketio.on('connect')
def handle_connect(auth):
    logging.info(f'Client connected: {threading.get_ident()}')


@socketio.on('disconnect')
def handle_disconnect(auth):
    logging.info(f'Client disconnected: {threading.get_ident()}')


if __name__ == '__main__':
    logging.info("Starting Flask-SocketIO server on http://localhost:5000")
    # Start the background task using SocketIO's helper
    socketio.start_background_task(target=stream_cameras_and_points)
    socketio.start_background_task(target=stream_camera_feeds)

    try:
        # Use socketio.run to start the server correctly
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, use_reloader=False, log_output=True)
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    finally:
        stop_event.set() # Signal the background thread to stop
