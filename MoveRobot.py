import naoqi
from naoqi import ALProxy
import math
RobotIP = "salvadors-macbook-pro.local."
RobotPort = 9559
#Sends processed frames to the robot
def sendrobot(robotIP="192.168.171.141", PORT=9559):
    try:
        try:
            motionProxy = ALProxy("ALMotion", robotIP, PORT) #creates proxy to call specific functions
        except Exception, e:
            print "Could not create proxy to AlMotion"
            print "Error was: ", e
        try:
            postureProxy = ALProxy("ALRobotPosture", robotIP, PORT) #creates proxy to call specific functions
        except Exception, e:
            print "Could not create proxy to ALRobotPosture"
            print "Error was: ", e

        # set RShoulderpitch to 0.5 radians (28.6478897565 degrees)
        angle = -119
        angle=math.radians(angle)
        motionProxy.setAngles("RShoulderPitch", angle, 0.5)
        postureProxy.goToPosture("StandInit", 0.5)
        angle = -76
        angle=math.radians(angle)
        motionProxy.setAngles("RShoulderRoll", angle, 0.5)
    except Exception, e:
        print "Could not create proxy to ALMotion"
        print "Error was: ", e

        
if __name__ == "__main__":
    sendrobot(RobotIP, RobotPort)