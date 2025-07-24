import numpy as np
import logging
import time
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from tracked_drone import TrackedDrone

class DroneTracker:
    """
    Central tracking system that manages multiple drone tracks.
    Handles data association, track creation, update, and deletion.
    """
    
    def __init__(self, 
                 max_drones: int = 4,
                 max_association_distance: float = 0.5,
                 track_timeout: float = 2.0,
                 cutoff_frequency: float = 15.0,
                 sampling_frequency: float = 60.0):
        """
        Initialize the drone tracker.
        
        Args:
            max_drones: Maximum number of drones to track simultaneously
            max_association_distance: Maximum distance for associating detections to tracks (meters)
            track_timeout: Time after which a track is considered stale (seconds)
            cutoff_frequency: Low-pass filter cutoff frequency (Hz)
            sampling_frequency: Expected sampling frequency (Hz)
        """
        self.max_drones = max_drones
        self.max_association_distance = max_association_distance
        self.track_timeout = track_timeout
        self.cutoff_frequency = cutoff_frequency
        self.sampling_frequency = sampling_frequency
        
        # Active tracks: track_id -> TrackedDrone
        self.tracked_drones: Dict[int, TrackedDrone] = {}
        self.next_track_id = 0
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.successful_associations = 0
        
        logging.info(f"DroneTracker initialized: max_drones={max_drones}, "
                    f"max_distance={max_association_distance}m, timeout={track_timeout}s")
    
    def update(self, detected_drones: List[Dict]) -> List[Dict]:
        """
        Update tracking with new detections.
        
        Args:
            detected_drones: List of detected drone dictionaries with 'center' and 'direction' keys
            
        Returns:
            List of filtered drone state dictionaries
        """
        self.frame_count += 1
        self.total_detections += len(detected_drones)
        
        # Convert detections to a more convenient format
        detections = []
        for drone_dict in detected_drones:
            center = np.array(drone_dict['center'])
            direction = np.array(drone_dict['direction'])
            
            # Calculate heading from direction vector
            heading = np.arctan2(direction[1], direction[0])
            
            detections.append({
                'position': center,
                'heading': heading
            })
        
        # Step 1: Predict all existing tracks
        predictions = {}
        for track_id, tracked_drone in self.tracked_drones.items():
            predicted_pos = tracked_drone.predict()
            predictions[track_id] = predicted_pos
        
        # Step 2: Associate detections to tracks
        associations, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, predictions
        )
        
        # Step 3: Update matched tracks
        updated_states = []
        for detection_idx, track_id in associations.items():
            detection = detections[detection_idx]
            tracked_drone = self.tracked_drones[track_id]
            
            # Update the track with the new measurement
            state = tracked_drone.update(detection['position'], detection['heading'])
            updated_states.append(state)
            
            self.successful_associations += 1
        
        # Step 4: Handle unmatched tracks (mark as missed)
        for track_id in unmatched_tracks:
            self.tracked_drones[track_id].miss()
            
            # Still include the predicted state for unmatched tracks (with reduced confidence)
            predicted_state = self.tracked_drones[track_id].get_predicted_state()
            if predicted_state['confidence'] > 0.3:  # Only include if confidence is reasonable
                updated_states.append(predicted_state)
        
        # Step 5: Create new tracks for unmatched detections
        if len(self.tracked_drones) < self.max_drones:
            for detection_idx in unmatched_detections:
                if len(self.tracked_drones) >= self.max_drones:
                    break
                    
                detection = detections[detection_idx]
                new_track = self._create_new_track(detection['position'], detection['heading'])
                
                # Add initial state for new track
                initial_state = new_track.get_last_state()
                updated_states.append(initial_state)
                
                logging.info(f"Created new track {new_track.track_id} at position {detection['position']}")
        
        # Step 6: Remove stale tracks
        self._remove_stale_tracks()
        
        # Log statistics periodically
        if self.frame_count % 300 == 0:  # Every 5 seconds at 60 FPS
            self._log_statistics()
        
        return updated_states
    
    def _associate_detections_to_tracks(self, 
                                      detections: List[Dict], 
                                      predictions: Dict[int, np.ndarray]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Associate detections to existing tracks using Hungarian algorithm.
        
        Args:
            detections: List of detection dictionaries
            predictions: Dictionary mapping track_id to predicted position
            
        Returns:
            Tuple of (associations dict, unmatched detection indices, unmatched track ids)
        """
        if not detections or not predictions:
            return {}, list(range(len(detections))), list(predictions.keys())
        
        # Build cost matrix: rows = detections, cols = tracks
        detection_positions = np.array([det['position'] for det in detections])
        track_ids = list(predictions.keys())
        predicted_positions = np.array([predictions[tid] for tid in track_ids])
        
        # Compute distance matrix
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        for i, det_pos in enumerate(detection_positions):
            for j, pred_pos in enumerate(predicted_positions):
                distance = np.linalg.norm(det_pos - pred_pos)
                cost_matrix[i, j] = distance
        
        # Apply distance threshold
        cost_matrix[cost_matrix > self.max_association_distance] = 1e6  # Large cost for invalid associations
        
        # Solve assignment problem
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = [], []
        
        # Extract valid associations
        associations = {}
        for det_idx, track_idx in zip(row_indices, col_indices):
            if cost_matrix[det_idx, track_idx] < self.max_association_distance:
                associations[det_idx] = track_ids[track_idx]
        
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(len(detections)) if i not in associations]
        unmatched_tracks = [tid for tid in track_ids if tid not in associations.values()]
        
        return associations, unmatched_detections, unmatched_tracks
    
    def _create_new_track(self, position: np.ndarray, heading: float) -> TrackedDrone:
        """
        Create a new track for an unmatched detection.
        
        Args:
            position: Initial position [x, y, z]
            heading: Initial heading in radians
            
        Returns:
            New TrackedDrone instance
        """
        track_id = self.next_track_id
        self.next_track_id += 1
        
        new_track = TrackedDrone(
            track_id=track_id,
            initial_position=position,
            initial_heading=heading,
            cutoff_frequency=self.cutoff_frequency,
            sampling_frequency=self.sampling_frequency
        )
        
        self.tracked_drones[track_id] = new_track
        return new_track
    
    def _remove_stale_tracks(self):
        """Remove tracks that have become stale."""
        stale_track_ids = []
        
        for track_id, tracked_drone in self.tracked_drones.items():
            if tracked_drone.is_stale(self.track_timeout):
                stale_track_ids.append(track_id)
        
        for track_id in stale_track_ids:
            logging.info(f"Removing stale track {track_id}")
            del self.tracked_drones[track_id]
    
    def get_active_tracks(self) -> List[Dict]:
        """
        Get all currently active track states.
        
        Returns:
            List of active track state dictionaries
        """
        return [drone.get_last_state() for drone in self.tracked_drones.values()]
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracked_drones)
    
    def reset(self):
        """Reset the tracker, removing all tracks."""
        logging.info("Resetting drone tracker")
        self.tracked_drones.clear()
        self.next_track_id = 0
        self.frame_count = 0
        self.total_detections = 0
        self.successful_associations = 0
    
    def _log_statistics(self):
        """Log tracking performance statistics."""
        association_rate = (self.successful_associations / max(self.total_detections, 1)) * 100
        
        logging.info(f"Tracking stats - Frame: {self.frame_count}, "
                    f"Active tracks: {len(self.tracked_drones)}, "
                    f"Association rate: {association_rate:.1f}%, "
                    f"Total detections: {self.total_detections}")
        
        # Log individual track info
        for track_id, tracked_drone in self.tracked_drones.items():
            state = tracked_drone.get_last_state()
            logging.debug(f"Track {track_id}: pos={state['position']}, "
                         f"conf={state['confidence']:.2f}, age={state['age']:.1f}s")
    
    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking performance metrics
        """
        association_rate = (self.successful_associations / max(self.total_detections, 1)) * 100
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.tracked_drones),
            'total_detections': self.total_detections,
            'successful_associations': self.successful_associations,
            'association_rate': association_rate,
            'track_ids': list(self.tracked_drones.keys())
        } 