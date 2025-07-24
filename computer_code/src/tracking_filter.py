import cv2
import numpy as np
import time
from typing import Optional

class TrackingFilter:
    """
    Kalman filter for tracking a single drone's state over time.
    
    State vector: [x, y, z, vx, vy, vz, ax, ay, az] (9D)
    Measurement vector: [x, y, z, heading] (4D)
    """
    
    def __init__(self, initial_position: np.ndarray, initial_heading: float = 0.0):
        """
        Initialize Kalman filter for drone tracking.
        
        Args:
            initial_position: Initial 3D position [x, y, z]
            initial_heading: Initial heading in radians
        """
        # State and measurement dimensions
        self.state_dim = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.measurement_dim = 3  # [x, y, z] - only position
        
        # Create OpenCV Kalman filter
        self.kalman = cv2.KalmanFilter(self.state_dim, self.measurement_dim)
        
        # Initialize state
        self.kalman.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.kalman.statePost[0:3, 0] = initial_position.astype(np.float32)
        self.kalman.statePre = self.kalman.statePost.copy()
        
        # Measurement matrix H - we observe position directly
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y  
            [0, 0, 1, 0, 0, 0, 0, 0, 0]   # z
        ], dtype=np.float32)
        
        # Initialize transition matrix (will be updated with actual dt later)
        dt = 1/60.0  # Default dt
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],         # x
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],         # y
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],         # z
            [0, 0, 0, 1, 0, 0, dt, 0, 0],                 # vx
            [0, 0, 0, 0, 1, 0, 0, dt, 0],                 # vy
            [0, 0, 0, 0, 0, 1, 0, 0, dt],                 # vz
            [0, 0, 0, 0, 0, 0, 1, 0, 0],                  # ax
            [0, 0, 0, 0, 0, 0, 0, 1, 0],                  # ay
            [0, 0, 0, 0, 0, 0, 0, 0, 1]                   # az
        ], dtype=np.float32)
        
        # Process noise covariance Q
        self.kalman.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * 1e-2
        
        # Measurement noise covariance R
        self.kalman.measurementNoiseCov = np.eye(self.measurement_dim, dtype=np.float32) * 1e-1
        
        # Initial error covariance
        self.kalman.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 1.0
        
        # State tracking
        self.prev_position = initial_position.copy()
        self.prev_time = time.time()
        self.heading = initial_heading
        self.last_update_time = time.time()
        
    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """
        Predict next state using motion model.
        
        Args:
            dt: Time step in seconds. If None, use default.
            
        Returns:
            Predicted position [x, y, z]
        """
        if dt is None:
            dt = 1/60.0  # Default 60 FPS
        
        # Update transition matrix with current dt
        # Constant acceleration motion model:
        # x_new = x + vx*dt + 0.5*ax*dt^2
        # vx_new = vx + ax*dt
        # ax_new = ax (constant)
        
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],         # x
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],         # y
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],         # z
            [0, 0, 0, 1, 0, 0, dt, 0, 0],                 # vx
            [0, 0, 0, 0, 1, 0, 0, dt, 0],                 # vy
            [0, 0, 0, 0, 0, 1, 0, 0, dt],                 # vz
            [0, 0, 0, 0, 0, 0, 1, 0, 0],                  # ax
            [0, 0, 0, 0, 0, 0, 0, 1, 0],                  # ay
            [0, 0, 0, 0, 0, 0, 0, 0, 1]                   # az
        ], dtype=np.float32)
        
        # Predict next state
        predicted_state = self.kalman.predict()
        return predicted_state[:3].flatten()  # Return predicted position
    
    def update(self, measurement: np.ndarray, heading: float) -> np.ndarray:
        """
        Update filter with new measurement.
        
        Args:
            measurement: Measured position [x, y, z]
            heading: Measured heading in radians
            
        Returns:
            Updated state position [x, y, z]
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update transition matrix with actual dt
        if dt > 0:
            self.kalman.transitionMatrix = np.array([
                [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],         # x
                [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],         # y
                [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],         # z
                [0, 0, 0, 1, 0, 0, dt, 0, 0],                 # vx
                [0, 0, 0, 0, 1, 0, 0, dt, 0],                 # vy
                [0, 0, 0, 0, 0, 1, 0, 0, dt],                 # vz
                [0, 0, 0, 0, 0, 0, 1, 0, 0],                  # ax
                [0, 0, 0, 0, 0, 0, 0, 1, 0],                  # ay
                [0, 0, 0, 0, 0, 0, 0, 0, 1]                   # az
            ], dtype=np.float32)
        
        # First predict the next state
        predicted_state = self.kalman.predict()
        
        # Create measurement vector [x, y, z]
        measurement_vector = measurement.astype(np.float32)
        
        # Then correct with the measurement
        corrected_state = self.kalman.correct(measurement_vector)
        
        # Update previous position for velocity computation
        self.prev_position = measurement.copy()
        
        # Store heading separately (simple filtering could be added here)
        self.heading = heading
        
        return corrected_state[:3].flatten()  # Return corrected position
    
    def get_state(self) -> dict:
        """
        Get current filtered state.
        
        Returns:
            Dictionary with position, velocity, acceleration, and heading
        """
        state = self.kalman.statePost.flatten()
        return {
            'position': state[:3],
            'velocity': state[3:6], 
            'acceleration': state[6:9],
            'heading': self.heading
        }
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.kalman.statePost[:3].flatten()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.kalman.statePost[3:6].flatten()
    
    def reset(self, position: np.ndarray, heading: float = 0.0):
        """
        Reset the filter with new initial conditions.
        
        Args:
            position: New initial position [x, y, z]
            heading: New initial heading in radians
        """
        self.kalman.statePost = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.kalman.statePost[0:3, 0] = position.astype(np.float32)
        self.kalman.statePre = self.kalman.statePost.copy()
        
        self.prev_position = position.copy()
        self.heading = heading
        self.prev_time = time.time()
        self.last_update_time = time.time()
        
        # Reset error covariance
        self.kalman.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 1.0 