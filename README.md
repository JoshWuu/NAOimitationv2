# NAOimitationv2
This project aims to use the Naoqi API and the Mediapipe computer vision library to enable the NAO robot to mimic human movements.

For the implementation of this project, I have relied on the code from https://zenodo.org/records/3935469 for the connection between the robot's movement and the computer vision part. I am continuously enhancing the code to ensure the robot can imitate movements more accurately.

I also relied on Nicolas Renotes tutorial of "AI Pose Estimation with Python and MediaPipe" the video can be found here: https://www.youtube.com/watch?v=05rM3GwxPZM. I used this tutorial to understand how to use the Mediapipe library and how to extract the joint angles from the human pose.



## Requirements
To run the code, the following must be installed on the computer:
* Python 2.7 and 3.8 or higher
* Numpy
* OpenCV
* Naoqi
* Mediapipe

## Execution
To execute the code, the robot must be connected to the same network as the computer, and the following command should be run:
```bash
python2 server.py
```
This will start a server on the computer that will receive joint angles from the robot and send them for the robot to move. Subsequently, on the robot, execute the following command:
```bash
python3 client.py
```
This will initiate a client on the robot responsible for sending joint angles to the computer so that it can forward them to the robot via the Naoqi API.

## Operation
The code operates as follows:
* The robot sends joint angles to the computer.
* The computer receives the angles and sends them to the robot.
* The robot receives the angles and moves accordingly.

## Extras
The files getAngles.py and MoveRobot.py can be executed separately to verify the robot's correct movement. The former is auxiliary for visualizing the robot's joint angles, and the latter is responsible for moving the robot using hardcoded angles.

## Enhancements
* Improve the robot's precision.
* Add more movements.

## Author
* **Salvador Costilla Caballero**