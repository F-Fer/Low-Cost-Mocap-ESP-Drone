import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

const PointMesh = ({ position }) => {
  const validPosition = position || [0, 0, 0];
  return (
    <mesh position={validPosition}>
      <sphereGeometry args={[0.05, 16, 16]} />
      <meshStandardMaterial color="red" />
    </mesh>
  );
};

const CameraMesh = ({ position, rotation }) => {
  const validPosition = position || [0, 0, 0];
  const validRotationMatrix = rotation && rotation.length === 3 && rotation[0].length === 3
    ? rotation
    : [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  const quaternion = useMemo(() => {
    const mat4 = new THREE.Matrix4();
    mat4.set(
      validRotationMatrix[0][0], validRotationMatrix[0][1], validRotationMatrix[0][2], 0,
      validRotationMatrix[1][0], validRotationMatrix[1][1], validRotationMatrix[1][2], 0,
      validRotationMatrix[2][0], validRotationMatrix[2][1], validRotationMatrix[2][2], 0,
      0, 0, 0, 1
    );
    return new THREE.Quaternion().setFromRotationMatrix(mat4);
  }, [validRotationMatrix]);

  const coneRotation = useMemo(() => new THREE.Euler(-Math.PI / 2, 0, 0), []);

  return (
    <group position={validPosition} quaternion={quaternion}>
      <mesh rotation={coneRotation}>
        <coneGeometry args={[0.1, 0.20, 8]} />
        <meshStandardMaterial color="cyan" wireframe={false} />
      </mesh>
      <axesHelper args={[0.2]} />
    </group>
  );
};

const DroneMesh = ({ position, direction }) => {
  const validPosition = position || [0, 0, 0];
  // Ensure direction is a valid THREE.Vector3, normalized
  const dirVector = useMemo(() => {
    if (direction && Array.isArray(direction) && direction.length === 3) {
      return new THREE.Vector3(direction[0], direction[1], direction[2]).normalize();
    }
    return new THREE.Vector3(0, 0, 1); // Default direction if invalid
  }, [direction]);

  const arrowColor = 0xff00ff; // Magenta color for the arrow

  return (
    <group position={validPosition}>
      {/* Drone body/center */}
      <mesh>
        <sphereGeometry args={[0.08, 16, 16]} /> {/* Slightly larger than points */}
        <meshStandardMaterial color="magenta" />
      </mesh>
      {/* Direction arrow */}
      <arrowHelper args={[dirVector, new THREE.Vector3(0,0,0), 0.5, arrowColor, 0.1, 0.05]} />
       {/* ArrowHelper args: (dir, origin, length, hex, headLength, headWidth) */}
    </group>
  );
};

const VisualizationCanvas = ({ points = [], cameras = [], drones = [] }) => {
  const handleScaleSystem = async () => {
    try {
      const response = await fetch('http://localhost:5000/set_scale_factor', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const result = await response.json();
      if (response.ok) {
        alert('Scale factor set successfully: ' + result.message);
        console.log('Scale factor set:', result);
      } else {
        alert('Failed to set scale factor: ' + result.error);
        console.error('Failed to set scale factor:', result.error);
      }
    } catch (error) {
      alert('Error setting scale factor: ' + error.message);
      console.error('Error setting scale factor:', error);
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <button
        onClick={handleScaleSystem}
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          zIndex: 1000, // Ensure button is on top of the canvas
          padding: '8px 12px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
        }}
      >
        Scale System with Drone
      </button>
      <Canvas style={{ background: '#111', height: '100%', width: '100%' }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} intensity={1.0} />
        <directionalLight position={[-5, 5, 5]} intensity={0.5} />

        {points.map(point => (
          <PointMesh key={point.id} position={point.position} />
        ))}

        {Array.isArray(cameras) && cameras.map(camera => (
          <CameraMesh key={camera.id} position={camera.position} rotation={camera.rotation} />
        ))}

        {/* Render Drones */}
        {Array.isArray(drones) && drones.map((drone, index) => (
          // Assuming drones might not have a unique ID from backend yet, use index as key
          <DroneMesh key={drone.id || `drone-${index}`} position={drone.position} direction={drone.direction} />
        ))}

        <OrbitControls />

        <axesHelper args={[5]} />
      </Canvas>
    </div>
  );
};

export default VisualizationCanvas;