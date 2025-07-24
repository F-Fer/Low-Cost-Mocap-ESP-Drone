// frontend/src/components/CameraFeeds.jsx
import React, { useState, useEffect } from 'react';

const captureCalibrationPoints = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/capture_calibration_points`, {
    method: 'POST'
  });
  const data = await response.json();
  return data;
}

const getNumCalibrationPoints = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/get_num_calibration_points`, {
    method: 'GET'
  });
  const data = await response.json();
  return { 
    numCalibrationPoints: data.num_calibration_points, 
    minCalibrationPoints: data.min_calibration_points 
  };
}

const clearCalibrationPoints = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/clear_calibration_points`, {
    method: 'POST',
  });
  const data = await response.json();
  return data;
}

const startCalibration = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/start_calibration`, {
    method: 'POST',
  });
  const data = await response.json();
  return data;
}

const setGroundHeightBackend = async (apiUrl, height) => {
  const response = await fetch(`${apiUrl}/set_ground_height`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ground_height: parseFloat(height) }),
  });
  const data = await response.json();
  return data;
};

const startCalibrationStream = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/start_calibration_stream`, {
    method: 'POST'
  });
  const data = await response.json();
  return data;
};

const stopCalibrationStream = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/stop_calibration_stream`, {
    method: 'POST'
  });
  const data = await response.json();
  return data;
};

const getCalibrationStreamStatus = async (apiUrl) => {
  const response = await fetch(`${apiUrl}/get_calibration_stream_status`, {
    method: 'GET'
  });
  const data = await response.json();
  return data;
};

const CameraFeeds = ({ numCameras, isConnected, cameraFrames, apiUrl, socket }) => {
  const [calibrationPoints, setCalibrationPoints] = useState({
    current: 0,
    required: 0,
    message: '',
    status: 'idle' // 'idle', 'success', 'error'
  });
  const [streamCalibration, setStreamCalibration] = useState({
    active: false,
    pointsCollected: 0,
    totalFramesProcessed: 0,
    collectionRate: 0,
    duration: 0,
    message: '',
    status: 'idle' // 'idle', 'streaming', 'success', 'error'
  });
  const [groundHeight, setGroundHeight] = useState('');
  const [groundHeightStatus, setGroundHeightStatus] = useState({
    message: '',
    status: 'idle' // 'idle', 'success', 'error'
  });

  useEffect(() => {
    const fetchNumCalibrationPoints = async () => {
      try {
        const result = await getNumCalibrationPoints(apiUrl);
        setCalibrationPoints(prev => ({
          ...prev,
          current: result.numCalibrationPoints,
          required: result.minCalibrationPoints
        }));
      } catch (error) {
        console.error("Error fetching calibration points:", error);
      }
    };
    
    if (isConnected) {
      fetchNumCalibrationPoints();
    }
  }, [apiUrl, isConnected]);

  // Socket.IO listeners for real-time stream updates
  useEffect(() => {
    if (socket) {
      const handleCalibrationStreamUpdate = (data) => {
        setStreamCalibration(prev => ({
          ...prev,
          pointsCollected: data.points_collected,
          totalFramesProcessed: data.total_frames_processed,
          collectionRate: data.collection_rate || 0,
          duration: data.duration || 0,
          message: data.valid_frame ? 
            `Collecting... (${data.points_collected} points, ${(data.collection_rate || 0).toFixed(1)} pts/s)` :
            `Processing frames... (${data.total_frames_processed} processed)`
        }));
      };

      socket.on('calibration_stream_update', handleCalibrationStreamUpdate);

      return () => {
        socket.off('calibration_stream_update', handleCalibrationStreamUpdate);
      };
    }
  }, [socket]);

  const handleCapturePoints = async () => {
    setCalibrationPoints(prev => ({ ...prev, status: 'idle', message: 'Capturing...' }));
    try {
      const result = await captureCalibrationPoints(apiUrl);
      if (result.error) {
        setCalibrationPoints(prev => ({ 
          ...prev, 
          status: 'error',
          message: result.error
        }));
      } else {
        const updatedCount = await getNumCalibrationPoints(apiUrl);
        setCalibrationPoints({
          current: updatedCount.numCalibrationPoints,
          required: updatedCount.minCalibrationPoints,
          status: 'success',
          message: result.message || 'Point captured successfully'
        });
      }
    } catch (error) {
      setCalibrationPoints(prev => ({ 
        ...prev, 
        status: 'error',
        message: 'Failed to capture point'
      }));
    }
  };

  const handleClearPoints = async () => {
    setCalibrationPoints(prev => ({ ...prev, status: 'idle', message: 'Clearing...' }));
    try {
      const result = await clearCalibrationPoints(apiUrl);
      const updatedCount = await getNumCalibrationPoints(apiUrl);
      setCalibrationPoints({
        current: updatedCount.numCalibrationPoints,
        required: updatedCount.minCalibrationPoints,
        status: 'success',
        message: result.message || 'Points cleared'
      });
    } catch (error) {
      setCalibrationPoints(prev => ({ 
        ...prev, 
        status: 'error',
        message: 'Failed to clear points'
      }));
    }
  };

  const handleStartCalibration = async () => {
    setCalibrationPoints(prev => ({ ...prev, status: 'idle', message: 'Starting calibration...' }));
    try {
      const result = await startCalibration(apiUrl);
      if (result.error) {
        setCalibrationPoints(prev => ({ 
          ...prev, 
          status: 'error',
          message: result.error
        }));
      } else {
        setCalibrationPoints(prev => ({
          ...prev,
          status: 'success',
          message: result.message || 'Calibration complete'
        }));
      }
    } catch (error) {
      setCalibrationPoints(prev => ({ 
        ...prev, 
        status: 'error',
        message: 'Calibration failed'
      }));
    }
  };

  const handleSetGroundHeight = async () => {
    if (groundHeight === '' || isNaN(parseFloat(groundHeight))) {
      setGroundHeightStatus({
        message: 'Please enter a valid number for ground height.',
        status: 'error'
      });
      return;
    }
    setGroundHeightStatus({ message: 'Setting ground height...', status: 'idle' });
    try {
      const result = await setGroundHeightBackend(apiUrl, groundHeight);
      if (result.error) {
        setGroundHeightStatus({
          message: result.error,
          status: 'error'
        });
      } else {
        setGroundHeightStatus({
          message: result.message || 'Ground height set successfully',
          status: 'success'
        });
      }
    } catch (error) {
      setGroundHeightStatus({
        message: 'Failed to set ground height',
        status: 'error'
      });
    }
  };

  const handleStartStreamCalibration = async () => {
    setStreamCalibration(prev => ({ 
      ...prev, 
      status: 'streaming', 
      message: 'Starting stream calibration...',
      pointsCollected: 0,
      totalFramesProcessed: 0,
      collectionRate: 0,
      duration: 0
    }));
    
    try {
      const result = await startCalibrationStream(apiUrl);
      if (result.error) {
        setStreamCalibration(prev => ({ 
          ...prev, 
          status: 'error',
          message: result.error,
          active: false
        }));
      } else {
        setStreamCalibration(prev => ({
          ...prev,
          status: 'streaming',
          message: 'Stream calibration active - move LED around!',
          active: true
        }));
      }
    } catch (error) {
      setStreamCalibration(prev => ({ 
        ...prev, 
        status: 'error',
        message: 'Failed to start stream calibration',
        active: false
      }));
    }
  };

  const handleStopStreamCalibration = async () => {
    setStreamCalibration(prev => ({ ...prev, message: 'Stopping stream calibration...' }));
    
    try {
      const result = await stopCalibrationStream(apiUrl);
      if (result.error) {
        setStreamCalibration(prev => ({ 
          ...prev, 
          status: 'error',
          message: result.error
        }));
      } else {
        setStreamCalibration(prev => ({
          ...prev,
          status: 'success',
          message: `Stream complete! Collected ${result.points_collected} points (${result.collection_rate.toFixed(1)} pts/s)`,
          active: false
        }));
        
        // Update main calibration points count
        const updatedCount = await getNumCalibrationPoints(apiUrl);
        setCalibrationPoints(prev => ({
          ...prev,
          current: updatedCount.numCalibrationPoints,
          required: updatedCount.minCalibrationPoints,
          status: 'success',
          message: `Total points: ${updatedCount.numCalibrationPoints}`
        }));
      }
    } catch (error) {
      setStreamCalibration(prev => ({ 
        ...prev, 
        status: 'error',
        message: 'Failed to stop stream calibration',
        active: false
      }));
    }
  };

  if (!isConnected) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-lg">Connect to the server to view camera feeds.</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full p-4 overflow-y-auto">
      <h2 className="text-xl font-semibold mb-4">Camera Feeds</h2>
      
      {/* Camera Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
        {Array.from({ length: numCameras }, (_, i) => {
          const cameraKey = String(i); 
          return (
            <div key={cameraKey} className="border border-gray-300 dark:border-gray-700 rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800">
              <div className="bg-gray-800 dark:bg-gray-900 text-white px-3 py-2">
                <span>Camera {i}</span> 
              </div>
              <div className="bg-black aspect-video flex items-center justify-center text-gray-400">
                {cameraFrames[cameraKey] ? ( 
                  <img 
                    src={`data:image/jpeg;base64,${cameraFrames[cameraKey]}`} 
                    alt={`Feed from camera ${i}`}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-4">
                    <p>Waiting for feed from Camera {i}...</p>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Stream Calibration Controls */}
      <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 mb-4 border border-gray-300 dark:border-gray-700">
        <h3 className="text-lg font-medium mb-4 text-gray-800 dark:text-gray-200">Stream Calibration (Recommended)</h3>
        
        {/* Stream Status Display */}
        <div className="mb-6">
          <div className="flex items-center mb-3">
            <span className="mr-2 font-medium">Status:</span>
            <span className={`px-3 py-1 rounded text-sm font-medium ${
              streamCalibration.status === 'streaming' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
              streamCalibration.status === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
              streamCalibration.status === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 
              'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }`}>
              {streamCalibration.message || 'Ready for stream calibration'}
            </span>
          </div>
          
          {streamCalibration.active && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded border">
                <div className="font-medium text-blue-600 dark:text-blue-400">Points Collected</div>
                <div className="text-xl font-bold">{streamCalibration.pointsCollected}</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border">
                <div className="font-medium text-blue-600 dark:text-blue-400">Collection Rate</div>
                <div className="text-xl font-bold">{streamCalibration.collectionRate.toFixed(1)} pts/s</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border">
                <div className="font-medium text-blue-600 dark:text-blue-400">Duration</div>
                <div className="text-xl font-bold">{streamCalibration.duration.toFixed(1)}s</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border">
                <div className="font-medium text-blue-600 dark:text-blue-400">Frames Processed</div>
                <div className="text-xl font-bold">{streamCalibration.totalFramesProcessed}</div>
              </div>
            </div>
          )}
        </div>
        
        {/* Stream Control Buttons */}
        <div className="flex gap-4">
          {!streamCalibration.active ? (
            <button 
              onClick={handleStartStreamCalibration}
              disabled={streamCalibration.status === 'streaming'}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white rounded transition-colors flex items-center justify-center"
            >
              <img src="/camera-video-fill.svg" className="h-5 w-5 mr-2 filter brightness-0 invert" alt="Camera video" />
              Start Stream Calibration
            </button>
          ) : (
            <button 
              onClick={handleStopStreamCalibration}
              className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded transition-colors flex items-center justify-center"
            >
              <img src="/camera-video-off-fill.svg" className="h-5 w-5 mr-2 filter brightness-0 invert" alt="Camera video off" />
              Stop Stream Calibration
            </button>
          )}
        </div>
        
        <div className="mt-4 text-sm text-blue-600 dark:text-blue-400">
          <p><strong>Instructions:</strong> Move a single bright LED around the camera field of view. The system will automatically collect calibration points when the LED is visible in all cameras.</p>
        </div>
      </div>

      {/* Manual Calibration Controls */}
      <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 mb-4">
        <h3 className="text-lg font-medium mb-4">Manual Calibration (Legacy)</h3>
        
        {/* Status Display */}
        <div className="mb-6 flex items-center">
          <div className="mr-4">
            <div className="flex items-center">
              <span className="mr-2 font-medium">Status:</span>
              <span className={`px-2 py-1 rounded text-sm ${
                calibrationPoints.status === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                calibrationPoints.status === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 
                'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
              }`}>
                {calibrationPoints.message || 'Ready'}
              </span>
            </div>
          </div>
          
          <div className="flex-grow">
            <div className="flex items-center">
              <div className="mr-3 font-medium">Collected Points:</div>
              <div className="w-full max-w-md bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                <div 
                  className={`h-2.5 rounded-full ${
                    calibrationPoints.current >= calibrationPoints.required 
                      ? 'bg-green-500' 
                      : 'bg-blue-500'
                  }`}
                  style={{ width: `${Math.min(100, (calibrationPoints.current / Math.max(1, calibrationPoints.required)) * 100)}%` }}
                ></div>
              </div>
              <span className="ml-3 font-mono">
                {calibrationPoints.current} / {calibrationPoints.required}
              </span>
            </div>
          </div>
        </div>
        
        {/* Control Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <button 
            onClick={handleCapturePoints}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <circle cx="12" cy="12" r="10" strokeWidth="2" />
              <circle cx="12" cy="12" r="3" strokeWidth="2" />
            </svg>
            Capture Point
          </button>
          
          <button 
            onClick={handleClearPoints}
            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded transition-colors flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear Points
          </button>
          
          <button 
            onClick={handleStartCalibration}
            disabled={calibrationPoints.current < calibrationPoints.required}
            className={`px-4 py-2 rounded transition-colors flex items-center justify-center ${
              calibrationPoints.current >= calibrationPoints.required
                ? 'bg-green-500 hover:bg-green-600 text-white'
                : 'bg-gray-300 text-gray-500 dark:bg-gray-700 dark:text-gray-400 cursor-not-allowed'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Start Calibration
          </button>
          
          <button 
            onClick={async () => {
              try {
                const result = await getNumCalibrationPoints(apiUrl);
                setCalibrationPoints(prev => ({
                  ...prev,
                  current: result.numCalibrationPoints,
                  required: result.minCalibrationPoints,
                  status: 'success',
                  message: 'Updated point count'
                }));
              } catch (error) {
                console.error("Error getting calibration points:", error);
                setCalibrationPoints(prev => ({ 
                  ...prev, 
                  status: 'error',
                  message: 'Failed to update point count'
                }));
              }
            }}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 rounded transition-colors flex items-center justify-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh Status
          </button>
        </div>
      </div>

      {/* Set Ground Height Controls */}
      <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 mb-4">
        <h3 className="text-lg font-medium mb-2">Set Ground Height</h3>
        <div className="flex items-center mb-3">
          <input 
            type="number"
            value={groundHeight}
            onChange={(e) => setGroundHeight(e.target.value)}
            placeholder="Enter ground height (e.g., 0.0)"
            className="flex-grow px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm dark:bg-gray-700 dark:text-white mr-3"
          />
          <button
            onClick={handleSetGroundHeight}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md transition-colors flex items-center justify-center"
          >
            Set Height
          </button>
        </div>
        {groundHeightStatus.message && (
          <div className={`text-sm ${groundHeightStatus.status === 'error' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
            {groundHeightStatus.message}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraFeeds;