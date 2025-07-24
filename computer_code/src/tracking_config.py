# Drone Tracking Configuration

# Tracking parameters
TRACKING_CONFIG = {
    # DroneTracker settings
    'max_drones': 4,  # Maximum number of drones to track simultaneously
    'max_association_distance': 0.5,  # Maximum distance for associating detections to tracks (meters)
    'track_timeout': 2.0,  # Time after which a track is considered stale (seconds)
    
    # Filter settings
    'cutoff_frequency': 15.0,  # Low-pass filter cutoff frequency (Hz)
    'sampling_frequency': 60.0,  # Expected sampling frequency (Hz)
    
    # Kalman filter parameters
    'kalman': {
        'process_noise': 1e-2,  # Process noise covariance scaling
        'measurement_noise': 1e-1,  # Measurement noise covariance scaling
        'initial_position_uncertainty': 1.0,  # Initial error covariance scaling
    },
    
    # Track quality parameters
    'track_quality': {
        'max_consecutive_misses': 10,  # Max missed detections before reducing confidence
        'min_confidence_threshold': 0.1,  # Minimum confidence to keep track alive
        'confidence_for_prediction': 0.3,  # Minimum confidence to include predicted states
    },
    
    # Low-pass filter parameters
    'low_pass': {
        'filter_order': 5,  # Butterworth filter order
        'buffer_size': 300,  # Size of filter buffer
    }
}

# Logging configuration for tracking
TRACKING_LOGGING = {
    'log_statistics_interval': 300,  # Frames between statistics logging (5 seconds at 60 FPS)
    'log_level': 'INFO',  # Logging level for tracking components
} 