import socket
import sys
import pickle
import math
import time
import argparse
from naoqi import ALProxy
import numpy as np

#Program to try and visualize the angles of the arms of a person and send them to the robot
# Global variables
listAngles = []
t = 0
# RobotIP = "192.168.171.141"
RobotIP = "salvadors-macbook-pro.local."
RobotPort = 9559
#Sends processed frames to the robot
def sendrobot(anglelist, robotIP="192.168.171.141", PORT=9559):
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

        global t # uses global variable t

        if (t == 0): # if it is the first time the robot is called upon
            motionProxy.setStiffnesses("Body", 0.6) # unstiffens the joints
            postureProxy.goToPosture("Crouch", 1.0) # gets the robot into his initial standing position

        names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
        #list of joints that will get changed
        n=len(names)
        angleLists = [[math.radians(anglelist[len(anglelist) - n])],
                        [math.radians(anglelist[len(anglelist) - n + 1])],
                        [math.radians(anglelist[len(anglelist) - n + 2])],
                        [math.radians(anglelist[len(anglelist) - n + 3])]
                        ]
        timeLists= [[0.5]]*n# sets the time the robot has to get to the joint location (when you give more than one coordinate for a joint, you have to give more than one timestamp for that same joint!)
        isAbsolute = True #  joint positions absolute and not relative
        print(angleLists)
        motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute) #the function= talks with the robot
        print "done"
        t += 1 
    except Exception: # checks for any and all errors
        pass # ignores every single one of them, except keyboardInterupt and SystemExit
        postureProxy.goToPosture("Crouch", 1.0) # set the robot in its initial position
        motionProxy.setStiffnesses("Body", 0.0) # stiffen the joints
    except (KeyboardInterrupt, SystemExit): # when the program gets terminated
        postureProxy.goToPosture("Crouch", 1.0) # set the robot in its initial position
        motionProxy.setStiffnesses("Body", 0.0) # stiffen the joints
        raise # quit



def ConnectionServer():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', 10000)
    print >>sys.stderr, 'starting up on %s port %s' % server_address
    sock.bind(server_address)


    sock.listen(1)

    connection, address = sock.accept()
    
    # Client conected
    print >>sys.stderr, 'cliente conectado'

    temp = 0

    while temp == 0:

        #recive json from client
        frames_json = connection.recv(4096)
        #convierte el json a lista de listas
        frames_list = pickle.loads(frames_json) # converts the frames from json to a list

        listAngles.append(frames_list[0])#RshoulderPitch
        listAngles.append(frames_list[1])#RshoulderRoll
        listAngles.append(frames_list[2])#RElbowRoll
        listAngles.append(frames_list[3])#RElbowYaw


        sendrobot(listAngles, RobotIP, RobotPort)

        if frames_json == 'fin':
            temp == 1
    connection.close()



if __name__ == '__main__':
    ConnectionServer()
    

