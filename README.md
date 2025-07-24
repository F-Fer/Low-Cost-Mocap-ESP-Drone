# Low Cost Mocap (for ESP-32 drones)

### A general purpose motion capture system built from the ground up, used to autonomously fly multiple drones indoors

This project is a fork of Joshua Birds awesome Low Cost Mocap project: [https://github.com/jyjblrd/Low-Cost-Mocap/tree/main](GitHub)
It expands the project by tracking affordabel esp32 drones and it does not need the open-cv sfm package. 

## YouTube Video
Watch this demo video!
[https://youtu.be/-AZVHd_bZnI](https://youtu.be/-AZVHd_bZnI)

[<img src="images/lcm-thumbnail.jpg">](https://youtu.be/0ql20JKrscQ?si=jkxyOe-iCG7fa5th)

## Dependencies
Install the pseyepy python library: [https://github.com/bensondaled/pseyepy](https://github.com/bensondaled/pseyepy)

This project **does not** require the sfm (structure from motion) OpenCV module. The calculations are all done using the standard open-cv python distribution.

install npm and nextJs

## Runing the code

From the computer_code/frontend directory run `npm install` to install node dependencies 

Then run `npm run dev` to start the webserver. You will be given a url view the frontend interface.

In another terminal window, run `python3 src/api.py` to start the backend server. This is what receives the camera streams and does motion capture computations.


