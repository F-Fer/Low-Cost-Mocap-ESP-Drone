'use client'; // Client side rendering

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic'; 
import { io } from 'socket.io-client'; 

// Backend API URL
const API_URL = 'http://localhost:5000';

// Dynamically import the canvas component
const VisualizationCanvas = dynamic(
  () => import('../components/VisualizationCanvas'),
  {
    ssr: false, // Ensure it's not rendered on the server
    loading: () => <p>Loading 3D Canvas...</p> // Optional loading state
  }
);

// Import camera feeds component
const CameraFeeds = dynamic(
  () => import('../components/CameraFeeds'),
  {
    ssr: false,
    loading: () => <p>Loading Camera Feeds...</p>
  }
);

// Tab options
const TABS = {
  LIVE_3D: 'LIVE_3D',
  CAMERA_FEEDS: 'CAMERA_FEEDS'
};

// Main Page Component
export default function Home() {
  const [points, setPoints] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [drones, setDrones] = useState([]);
  const [cameraFrames, setCameraFrames] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState(TABS.LIVE_3D);
  const socketRef = useRef(null);

  useEffect(() => {
    // Get the origin of the current page
    const origin = window.location.hostname;
    const port = 5000;  

    // Initialize socket connection
    const socket = io(`${origin}:${port}`, {
        transports: ['websocket'],
        reconnectionAttempts: 5, // Limit reconnection attempts
    });
    socketRef.current = socket; // Store socket in ref

    // --- Socket Event Listeners ---
    socket.on('connect', () => {
      console.log('Socket connected:', socket.id);
      setIsConnected(true);
    });

    socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      setIsConnected(false);
    });

    socket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        setIsConnected(false);
    });

    // Listen for the 'update_points' event from the backend
    socket.on('update_points', (data) => {
      if (data && data.points) {
        setPoints(data.points); // Update the state with new points
      }
      if (data && data.cameras) {
        setCameras(data.cameras); // Update the state with new cameras
      }
      if (data && data.drones) {
        setDrones(data.drones); // Update the state with new drones
      }
    });

    // Listen for camera frames
    socket.on('camera_frame', (data) => {
      if (!data || !('camera_id' in data) || !data.frame) return;
      const id = String(data.camera_id);
      setCameraFrames(prev => ({ ...prev, [id]: data.frame }));
    });

    // --- Cleanup Function ---
    // This runs when the component unmounts
    return () => {
      console.log('Disconnecting socket...');
      socket.disconnect();
      socketRef.current = null;
      setIsConnected(false);
    };
  }, []);

  const renderTabContent = () => {
    switch (activeTab) {
      case TABS.LIVE_3D:
        return isConnected ? (
          <VisualizationCanvas points={points} cameras={cameras} drones={drones} />
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-lg">Attempting to connect to the data server...</p>
          </div>
        );
      case TABS.CAMERA_FEEDS:
        return <CameraFeeds numCameras={cameras.length} isConnected={isConnected} cameraFrames={cameraFrames} apiUrl={API_URL} socket={socketRef.current} />;
      default:
        return null;
    }
  };

  return (
    <main className="flex flex-col h-screen">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-300 bg-opacity-60 backdrop-blur-sm">
        <button 
          onClick={() => setActiveTab(TABS.LIVE_3D)}
          className={`px-6 py-3 border-b-2 transition-colors ${
            activeTab === TABS.LIVE_3D 
              ? 'border-blue-500 font-medium'
              : 'border-transparent hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          Live 3D
        </button>
        <button 
          onClick={() => setActiveTab(TABS.CAMERA_FEEDS)}
          className={`px-6 py-3 border-b-2 transition-colors ${
            activeTab === TABS.CAMERA_FEEDS 
              ? 'border-blue-500 font-medium'
              : 'border-transparent hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          Camera Feeds
        </button>
      </div>
      
      {/* Status Bar */}
      <div className="px-4 py-2 flex justify-end text-sm border-b border-gray-200 dark:border-gray-800">
        <span>
          Status: <span className={isConnected ? 'text-green-500' : 'text-red-500'}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </span>
      </div>
      
      {/* Tab Content Container */}
      <div className="flex-grow w-full relative">
        {renderTabContent()}
      </div>
    </main>
  );
}