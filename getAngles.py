
import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import time
import pickle
from threading import Thread

#Program to try and visualize the angles of the arms of a person


def angleRShoulderPitch(Rshoulder,Relbow): #calulates the Shoulderpitch value for the Left shoulder by using geometry
    x2 = Rshoulder[0]
    y2 = Rshoulder[1]
    z2 = Rshoulder[2]
    x1 = Relbow[0]
    y1 = Relbow[1]
    z1 = Relbow[2]
    if(y2>y1):
        angle = math.atan(abs(y2 - y1) / abs(z2 - z1)) 
        angle = math.degrees(angle)
        angle = -(angle)
        if(angle<-118):
            angle = -117
        return angle
    else:
        angle = math.atan((z2-z1)/(y1-y2))
        angle = math.degrees(angle)
        angle = 90-angle
        return angle


def angleRShoulderRoll(Rshoulder,Relbow): #calulates the ShoulderRoll value for the Right shoulder by using geometry
    x2 = Rshoulder[0]
    y2 = Rshoulder[1]
    z2 = Rshoulder[2]
    x1 = Relbow[0]
    y1 = Relbow[1]
    z1 = Relbow[2]
    if(z2<z1):
        test = z2
        anderetest = z1
        z2=anderetest
        z1=test
    if (z2 - z1 < 0.1):
        z2 = 1.0
        z1 = 0.8
    angle = math.atan((x1 - x2) / (z2 - z1))
    angle = math.degrees(angle)
    return angle

def angleRElbowRoll(Rshoulder,Relbow,Rwrist): #calulates the ElbowRoll value for the Right elbow by using geometry
    x3 = Rshoulder[0]
    y3 = Rshoulder[1]
    z3 = Rshoulder[2]
    x2 = Relbow[0]
    y2 = Relbow[1]
    z2 = Relbow[2]
    x1 = Rwrist[0]
    y1 = Rwrist[1]
    z1 = Rwrist[2]


    a1=(x3-x2)**2+(y3-y2)**2 + (z3-z2)**2 
    lineA= a1 ** 0.5                        # calculates length of line between 2 3D coordinates
    b1=(x2-x1)**2+(y2-y1)**2 + (z2-z1)**2
    lineB= b1 ** 0.5                        # calculates length of line between 2 3D coordinates
    c1=(x1-x3)**2+(y1-y3)**2 + (z1-z3)**2
    lineC= c1 ** 0.5                        # calculates length of line between 2 3D coordinates

    cosB = (pow(lineA, 2) + pow(lineB,2) - pow(lineC,2))/(2*lineA*lineB)
    acosB = math.acos(cosB)
    angle = math.degrees(acosB)
    angle = 180 - angle
    return angle

def angleRElbowYaw(Relbow,Rwrist,shoulderpitch): #calulates the ElbowYaw value for the Right elbow by using geometry
    x2 = Relbow[0]
    y2 = Relbow[1] 
    z2 = Relbow[2]
    x1 = Rwrist[0]
    y1 = Rwrist[1]
    z1 = Rwrist[2]

    if(abs(y2-y1)<0.2 and abs(z2-z1) < 0.2 and (x1>x2) ):
        return 0
    elif(abs(x2-x1)<0.1 and abs(z2-z1)<0.1 and (y1<y2)):
        return 90
    elif(abs(x2-x1)<0.1 and abs(z2-z1)<0.1 and (shoulderpitch > 50)):
        return 90
    elif(abs(y2-y1)<0.1 and abs(z2-z1)<0.1 and (shoulderpitch < 50)):
        return 0
    elif(abs(x2-x1)<0.1 and abs(y2-y1)<0.1 and (shoulderpitch > 50)):
        return 90
    else:
        angle = math.atan((z2 - z1) / (y1 - y2))
        angle = math.degrees(angle)
        angle = - angle + (shoulderpitch)
        angle = - angle
        return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

#12 = right shoulder
#14 = right elbow
#16 = right wrist

#13 = left shoulder
#15 = left elbow
#17 = left wrist


# change the color for the labels black
colorlabels = (0, 0, 0)



## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Flip the frame horizontally
        # frame = cv2.flip(frame, 1)
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates of the left arm
            Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z ]
            Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]


            # Get coordinates of the right arm
            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z] 
            Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

            # # visualize coordenates of right shoulder
            # cv2.putText(image, f"Right Shoulder: {Rshoulder[0]:.2f}, {Rshoulder[1]:.2f} {Rshoulder[2]:.2f}",
            #             tuple(np.multiply([Rshoulder[0], Rshoulder[1]], [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
 




            # calculate angle RShoulderPitch
            angleRShoulderPitchDegrees = angleRShoulderPitch(Rshoulder, Relbow)
            # Visualize angle with bigger letters
            cv2.putText(image, f"Right Shoulder Pitch: {angleRShoulderPitchDegrees:.2f} degrees", 
            tuple(np.multiply([Rshoulder[0], Rshoulder[1]], [640, 480]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)

            # calculate angle RShoulderRoll
            angleRShoulderRollDegrees = angleRShoulderRoll(Rshoulder, Relbow)
            # Visualize angle with bigger letters
            cv2.putText(image, f"Right ShoulderRoll: {angleRShoulderRollDegrees:.2f} degrees",
                        tuple(np.multiply([Rshoulder[0], Rshoulder[1]], [640, 580]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
             # # calculate angle RElbowRoll
            angleRElbowRollDegrees = angleRElbowRoll(Rshoulder, Relbow, Rwrist)
            # Visualize angle with bigger letters
            cv2.putText(image, f"Right ElbowRoll: {angleRElbowRollDegrees:.2f} degrees",
                        tuple(np.multiply([Relbow[0], Relbow[1]], [640, 680]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
            # calculate angle RElbowYaw
            angleRElbowYawDegrees = angleRElbowYaw(Relbow,Rwrist, angleRShoulderPitchDegrees)
            # Visualize angle with bigger letters
            cv2.putText(image, f"Right ElbowYaw: {angleRElbowYawDegrees:.2f} degrees",
                        tuple(np.multiply([Relbow[0], Relbow[1]], [640, 780]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
           



        except Exception as e:
            print(e)
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
