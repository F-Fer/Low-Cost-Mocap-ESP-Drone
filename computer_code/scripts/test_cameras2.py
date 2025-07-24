from pseyepy import Camera
import cv2
import numpy as np
import json 
import time

NUM_CAMERAS = 4
FPS = 60
THRESHOLD_VALUE = 40
EXPOSURE = 250
GAIN = 30

def make_square(img):
    x, y, _ = img.shape
    size = max(x, y)
    new_img = np.zeros((size, size, 3), dtype=np.uint8)
    ax,ay = (size - img.shape[1])//2,(size - img.shape[0])//2
    new_img[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img

    # Pad the new_img array with edge pixel values
    # Apply feathering effect
    feather_pixels = 8
    for i in range(feather_pixels):
        alpha = (i + 1) / feather_pixels
        new_img[ay - i - 1, :] = img[0, :] * (1 - alpha)  # Top edge
        new_img[ay + img.shape[0] + i, :] = img[-1, :] * (1 - alpha)  # Bottom edge

    return new_img

# Load camera parameters
f = open("/Users/finnferchau/dev/datfus/camera-params.json")
camera_params = json.load(f)
KS = [np.array(camera_params[i]["intrinsic_matrix"]) for i in range(NUM_CAMERAS)]
DISTROTION_COEFFICIENTS = [np.array(camera_params[i]["distortion_coef"]) for i in range(NUM_CAMERAS)]
ROTATIONS = [np.array(camera_params[i]["rotation"]) for i in range(NUM_CAMERAS)]


cameras = Camera(fps=FPS, resolution=Camera.RES_SMALL, gain=GAIN, exposure=EXPOSURE, colour=True)

kernel = np.array([[-2,-1,-1,-1,-2],
                        [-1,1,3,1,-1],
                        [-1,3,4,3,-1],
                        [-1,1,3,1,-1],
                        [-2,-1,-1,-1,-2]])

while True:
    frames, _ = cameras.read(squeeze=False)
    points_per_camera = [] # [num_cameras, 2]
    for i, frame in enumerate(frames):
        # Iterate over cameras
        frame = np.rot90(frame, k=ROTATIONS[i])
        frame = make_square(frame)
        frame = cv2.undistort(frame, KS[i], DISTROTION_COEFFICIENTS[i])
        frame = cv2.GaussianBlur(frame,(9,9),0)
        frame = cv2.filter2D(frame, -1, kernel)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames[i] = frame

    # Display frames side by side
    combined_frame = np.hstack(frames[:NUM_CAMERAS])
    cv2.imshow('PS3i Cameras', combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1/FPS)

cv2.destroyAllWindows()