import numpy as np
import time
from typing import Optional
from tracking_filter import TrackingFilter
from low_pass_filter import LowPassFilter

class TrackedDrone:
    """
    Represents a tracked drone with state history and filtering.
    Combines Kalman filtering with low-pass filtering for smooth motion.
    """
    
    def __init__(self, 
                 track_id: int, 
                 initial_position: np.ndarray, 
                 initial_heading: float = 0.0,
                 cutoff_frequency: float = 15.0,
                 sampling_frequency: float = 60.0):
        """
        Initialize tracked drone.
        
        Args:
            track_id: Unique identifier for this track
            initial_position: Initial 3D position [x, y, z]
            initial_heading: Initial heading in radians
            cutoff_frequency: Low-pass filter cutoff frequency in Hz
            sampling_frequency: Expected sampling frequency in Hz
        """
        self.track_id = track_id
        self.creation_time = time.time()
        self.last_update_time = time.time()
        self.update_count = 0
        
        # Core tracking filter
        self.tracking_filter = TrackingFilter(initial_position, initial_heading)
        
        # Low-pass filters for additional smoothing
        self.position_filter_xy = LowPassFilter(cutoff_frequency, sampling_frequency, dims=2)
        self.position_filter_z = LowPassFilter(cutoff_frequency, sampling_frequency, dims=1)
        self.velocity_filter_xy = LowPassFilter(cutoff_frequency, sampling_frequency, dims=2)
        self.velocity_filter_z = LowPassFilter(cutoff_frequency, sampling_frequency, dims=1)
        self.heading_filter = LowPassFilter(cutoff_frequency, sampling_frequency, dims=1)
        
        # Track quality metrics
        self.confidence = 1.0
        self.consecutive_misses = 0
        self.max_consecutive_misses = 10
        
        # Store filtered state
        self._last_filtered_state = {
            'position': initial_position.copy(),
            'velocity': np.zeros(3),
            'acceleration': np.zeros(3),
            'heading': initial_heading,
            'track_id': track_id,
            'confidence': self.confidence,
            'age': 0.0
        }
    
    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Predict next position using Kalman filter.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Predicted position [x, y, z]
        """
        return self.tracking_filter.predict(dt)
    
    def update(self, position: np.ndarray, heading: float) -> dict:
        """
        Update tracking with new measurement.
        
        Args:
            position: Measured 3D position [x, y, z]
            heading: Measured heading in radians
            
        Returns:
            Filtered drone state dictionary
        """
        self.last_update_time = time.time()
        self.update_count += 1
        self.consecutive_misses = 0
        
        # Update Kalman filter
        kalman_position = self.tracking_filter.update(position, heading)
        kalman_state = self.tracking_filter.get_state()
        
        # Apply additional low-pass filtering
        filtered_pos_xy = self.position_filter_xy.filter(kalman_state['position'][:2])
        filtered_pos_z = self.position_filter_z.filter(kalman_state['position'][2])
        filtered_position = np.concatenate([filtered_pos_xy, [filtered_pos_z]])
        
        filtered_vel_xy = self.velocity_filter_xy.filter(kalman_state['velocity'][:2])
        filtered_vel_z = self.velocity_filter_z.filter(kalman_state['velocity'][2])
        filtered_velocity = np.concatenate([filtered_vel_xy, [filtered_vel_z]])
        
        filtered_heading = self.heading_filter.filter(heading)
        
        # Update confidence based on tracking quality
        self._update_confidence()
        
        # Store and return filtered state
        self._last_filtered_state = {
            'position': filtered_position,
            'velocity': filtered_velocity,
            'acceleration': kalman_state['acceleration'],
            'heading': filtered_heading,
            'track_id': self.track_id,
            'confidence': self.confidence,
            'age': self.last_update_time - self.creation_time
        }
        
        return self._last_filtered_state.copy()
    
    def miss(self):
        """Mark a detection miss (no matching measurement this frame)."""
        self.consecutive_misses += 1
        self._update_confidence()
    
    def is_stale(self, timeout: float = 2.0) -> bool:
        """
        Check if track is stale and should be removed.
        
        Args:
            timeout: Maximum time since last update in seconds
            
        Returns:
            True if track should be removed
        """
        time_since_update = time.time() - self.last_update_time
        return (time_since_update > timeout or 
                self.consecutive_misses > self.max_consecutive_misses or
                self.confidence < 0.1)
    
    def get_last_state(self) -> dict:
        """Get the last filtered state."""
        return self._last_filtered_state.copy()
    
    def get_predicted_state(self, dt: float = 1/60.0) -> dict:
        """
        Get predicted state for next time step.
        
        Args:
            dt: Time step for prediction
            
        Returns:
            Predicted state dictionary
        """
        predicted_position = self.predict(dt)
        current_state = self.get_last_state()
        
        # Simple linear prediction for other quantities
        predicted_state = current_state.copy()
        predicted_state['position'] = predicted_position
        predicted_state['confidence'] *= 0.9  # Reduce confidence for predictions
        
        return predicted_state
    
    def _update_confidence(self):
        """Update tracking confidence based on recent performance."""
        # Base confidence on consecutive misses
        miss_penalty = min(self.consecutive_misses / self.max_consecutive_misses, 1.0)
        
        # Base confidence on track age (newer tracks are less reliable)
        age_factor = min(self.update_count / 10.0, 1.0)  # Full confidence after 10 updates
        
        # Combine factors
        self.confidence = (1.0 - miss_penalty) * age_factor
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def reset_filters(self):
        """Reset all filters (useful for re-initialization)."""
        self.position_filter_xy.reset()
        self.position_filter_z.reset()
        self.velocity_filter_xy.reset()
        self.velocity_filter_z.reset()
        self.heading_filter.reset()
        
        # Reset tracking filter to current position
        current_pos = self._last_filtered_state['position']
        current_heading = self._last_filtered_state['heading']
        self.tracking_filter.reset(current_pos, current_heading) 