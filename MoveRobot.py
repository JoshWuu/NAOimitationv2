# import naoqi
# from naoqi import ALProxy
# import math
# #program to see the angles in the robot
# RobotIP = "salvadors-macbook-pro.local."
# RobotPort = 9559
# #Sends processed frames to the robot
# def sendrobot(robotIP="192.168.171.141", PORT=9559):
#     try:
#         try:
#             motionProxy = ALProxy("ALMotion", robotIP, PORT) #creates proxy to call specific functions
#         except Exception, e:
#             print "Could not create proxy to AlMotion"
#             print "Error was: ", e
#         try:
#             postureProxy = ALProxy("ALRobotPosture", robotIP, PORT) #creates proxy to call specific functions
#         except Exception, e:
#             print "Could not create proxy to ALRobotPosture"
#             print "Error was: ", e

#         # set RShoulderpitch to 0.5 radians (28.6478897565 degrees)
#         angle = -119
#         angle=math.radians(angle)
#         motionProxy.setAngles("RShoulderPitch", angle, 0.5)
#         postureProxy.goToPosture("StandInit", 0.5)
#         angle = -76
#         angle=math.radians(angle)
#         motionProxy.setAngles("RShoulderRoll", angle, 0.5)
#     except Exception, e:
#         print "Could not create proxy to ALMotion"
#         print "Error was: ", e

        
# if __name__ == "__main__":
#     sendrobot(RobotIP, RobotPort)

#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use transformInterpolations Method on Arm"""

import qi
import argparse
import sys
import motion
# import almath
import math



def main(session):
    """
    Use case of transformInterpolations API.
    """
    # Get the services ALMotion & ALRobotPosture.

    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # Wake up robot
    motion_service.wakeUp()

    # Send robot to Stand Init
    posture_service.goToPosture("StandInit", 0.5)

    effector   = "LArm"
    frame      = motion.FRAME_TORS
    axisMask   = almath.AXIS_MASK_VEL # just control position
    useSensorValues = False

    path = []
    currentTf = motion_service.getTransform(effector, frame, useSensorValues)
    targetTf  = almath.Transform(currentTf)
    targetTf.r1_c4 += 0.03 # x
    targetTf.r2_c4 += 0.03 # y

    path.append(list(targetTf.toVector()))
    path.append(currentTf)

    # Go to the target and back again
    times      = [2.0, 4.0] # seconds

    motion_service.transformInterpolations(effector, frame, path, axisMask, times)

    # Go to rest position
    motion_service.rest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.171.148",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)