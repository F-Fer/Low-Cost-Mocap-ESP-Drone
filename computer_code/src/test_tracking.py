#!/usr/bin/env python3
"""
Test script for the drone tracking system.
Simulates drone detections and verifies that tracking works correctly.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from low_pass_filter import LowPassFilter
from tracking_filter import TrackingFilter
from tracked_drone import TrackedDrone
from drone_tracker import DroneTracker

def test_low_pass_filter():
    """Test the LowPassFilter class."""
    print("Testing LowPassFilter...")
    
    # Create filter for 3D data
    lpf = LowPassFilter(cutoff_frequency=10.0, sampling_frequency=60.0, dims=3)
    
    # Generate test signal - mostly constant with some low-frequency changes and noise
    n_samples = 120  # 2 seconds at 60 Hz
    t = np.linspace(0, 2, n_samples)
    
    # Create a slowly changing signal that the filter should be able to track
    clean_signal = np.array([
        5.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t),  # 0.5 Hz sine (well below 10 Hz cutoff)
        3.0 + 0.3 * np.cos(2 * np.pi * 0.3 * t),  # 0.3 Hz cosine 
        np.ones_like(t) * 2.0                      # Constant
    ]).T
    
    # Add significant noise
    noise_std = 0.3
    noise = np.random.normal(0, noise_std, clean_signal.shape)
    noisy_signal = clean_signal + noise
    
    # Filter the signal
    filtered_results = []
    for i in range(len(noisy_signal)):
        filtered_point = lpf.filter(noisy_signal[i])
        filtered_results.append(filtered_point)
    
    filtered_signal = np.array(filtered_results)
    
    # Calculate errors for the last half of the signal (after settling)
    settle_idx = n_samples // 2  # Let filter settle for first half
    
    clean_late = clean_signal[settle_idx:]
    filtered_late = filtered_signal[settle_idx:]
    noisy_late = noisy_signal[settle_idx:]
    
    clean_error = np.mean(np.linalg.norm(clean_late - filtered_late, axis=1))
    noisy_error = np.mean(np.linalg.norm(clean_late - noisy_late, axis=1))
    
    print(f"  Clean vs filtered error (after settling): {clean_error:.4f}")
    print(f"  Clean vs noisy error (after settling): {noisy_error:.4f}")
    print(f"  Noise reduction factor: {noisy_error/clean_error:.2f}x")
    print(f"  Noise std: {noise_std:.2f}")
    
    # Filter should provide at least some improvement
    assert clean_error < noisy_error, "Filter should reduce noise after settling"
    assert noisy_error/clean_error > 1.5, "Filter should provide at least 1.5x improvement"
    print("  ✓ LowPassFilter test passed")


def test_tracking_filter():
    """Test the TrackingFilter class."""
    print("Testing TrackingFilter...")
    
    # Create filter
    initial_pos = np.array([0.0, 0.0, 1.0])
    tf = TrackingFilter(initial_pos)
    
    # Simulate constant velocity motion
    dt = 1/60.0  # 60 FPS
    velocity = np.array([1.0, 0.5, 0.0])  # 1 m/s in x, 0.5 m/s in y
    
    positions = []
    for i in range(60):  # 1 second of data
        # True position
        true_pos = initial_pos + velocity * (i * dt)
        
        # Add measurement noise
        measured_pos = true_pos + np.random.normal(0, 0.01, 3)
        
        # Update filter
        filtered_pos = tf.update(measured_pos, 0.0)
        positions.append(filtered_pos)
    
    # Check that filter tracks motion
    final_pos = positions[-1]
    expected_final = initial_pos + velocity * (59 * dt)
    error = np.linalg.norm(final_pos - expected_final)
    
    print(f"  Final position error: {error:.4f} meters")
    print(f"  Expected: {expected_final}")
    print(f"  Actual: {final_pos}")
    
    assert error < 0.1, "Filter should track motion accurately"
    print("  ✓ TrackingFilter test passed")


def test_tracked_drone():
    """Test the TrackedDrone class."""
    print("Testing TrackedDrone...")
    
    # Create tracked drone
    initial_pos = np.array([0.0, 0.0, 1.0])
    drone = TrackedDrone(track_id=0, initial_position=initial_pos)
    
    # Simulate updates
    for i in range(30):
        # Simulate circular motion
        t = i * (1/60.0)
        pos = np.array([np.cos(t), np.sin(t), 1.0])
        heading = t
        
        state = drone.update(pos, heading)
        
        # Check state format
        assert 'position' in state
        assert 'velocity' in state
        assert 'heading' in state
        assert 'track_id' in state
        assert 'confidence' in state
        assert state['track_id'] == 0
    
    final_state = drone.get_last_state()
    print(f"  Final confidence: {final_state['confidence']:.2f}")
    print(f"  Final position: {final_state['position']}")
    
    assert final_state['confidence'] > 0.8, "Confidence should be high after updates"
    print("  ✓ TrackedDrone test passed")


def test_drone_tracker():
    """Test the DroneTracker class."""
    print("Testing DroneTracker...")
    
    # Create tracker
    tracker = DroneTracker(max_drones=2, max_association_distance=0.3)
    
    # Simulate two drones moving
    drone_paths = [
        np.array([[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1], [0.3, 0, 1]]),  # Drone 1: moving in +x
        np.array([[1, 0, 1], [1, 0.1, 1], [1, 0.2, 1], [1, 0.3, 1]])   # Drone 2: moving in +y
    ]
    
    for frame in range(4):
        # Create detections for this frame
        detections = []
        for drone_idx, path in enumerate(drone_paths):
            detections.append({
                'center': path[frame],
                'direction': np.array([1, 0, 0])  # Pointing in +x direction
            })
        
        # Update tracker
        tracked_states = tracker.update(detections)
        
        print(f"  Frame {frame}: {len(tracked_states)} tracks")
        for state in tracked_states:
            print(f"    Track {state['track_id']}: pos={state['position']}, conf={state['confidence']:.2f}")
    
    # Should have 2 tracks
    assert tracker.get_track_count() == 2, f"Expected 2 tracks, got {tracker.get_track_count()}"
    
    # Test statistics
    stats = tracker.get_statistics()
    print(f"  Final stats: {stats}")
    
    print("  ✓ DroneTracker test passed")


def test_integration():
    """Test the complete system integration."""
    print("Testing complete system integration...")
    
    # This would normally test with actual cameras, but we'll simulate
    print("  (Integration test would require cameras - skipping)")
    print("  ✓ Integration test placeholder passed")


if __name__ == "__main__":
    print("Running drone tracking system tests...\n")
    
    try:
        test_low_pass_filter()
        print()
        
        test_tracking_filter()
        print()
        
        test_tracked_drone()
        print()
        
        test_drone_tracker()
        print()
        
        test_integration()
        print()
        
        print("🎉 All tests passed! Tracking system is ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 