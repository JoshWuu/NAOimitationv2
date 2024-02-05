# Client that sends the angles to the server in a list of lists format
# it also displays the angles on the screen and it can read the camera or a video reading the arguments from the terminal
import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import time
import pickle
from threading import Thread
import sys
#12 = right shoulder
#14 = right elbow
#16 = right wrist

#13 = left shoulder
#15 = left elbow
#17 = left wrist


#program that gets the angles and sends them to the server to send them to the robot

# change the color for the labels black
colorlabels = (0, 0, 0)
# right arm
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

# left arm
def angleLShoulderPitch(Lshoulder,Lelbow): #calulates the Shoulderpitch value for the Left shoulder by using geometry
    x2 = Lshoulder[0]
    y2 = Lshoulder[1]
    z2 = Lshoulder[2]
    x1 = Lelbow[0]
    y1 = Lelbow[1]
    z1 = Lelbow[2]
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


def angleLShoulderRoll(Lshoulder,Lelbow): #calulates the ShoulderRoll value for the Right shoulder by using geometry
    x2 = Lshoulder[0]
    y2 = Lshoulder[1]
    z2 = Lshoulder[2]
    x1 = Lelbow[0]
    y1 = Lelbow[1]
    z1 = Lelbow[2]
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

def angleLElbowRoll(Lshoulder,Lelbow,Lwrist): #calulates the ElbowRoll value for the Right elbow by using geometry
    x3 = Lshoulder[0]
    y3 = Lshoulder[1]
    z3 = Lshoulder[2]
    x2 = Lelbow[0]
    y2 = Lelbow[1]
    z2 = Lelbow[2]
    x1 = Lwrist[0]
    y1 = Lwrist[1]
    z1 = Lwrist[2]


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

def angleLElbowYaw(Lelbow,Lwrist,shoulderpitch): #calulates the ElbowYaw value for the Right elbow by using geometry
    x2 = Lelbow[0]
    y2 = Lelbow[1] 
    z2 = Lelbow[2]
    x1 = Lwrist[0]
    y1 = Lwrist[1]
    z1 = Lwrist[2]

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
def angleHeadYaw(head,Rshoulder,Lshoulder):
    # middle of the shoulders
    x1 = (Rshoulder[0] + Lshoulder[0]) / 2
    z1 = (Rshoulder[2] + Lshoulder[2]) / 2
    # coordinates of the head
    x2 = head[0]
    z2 = head[2]
    #vector in z direction
    V1=[0,0,-10]
    # vector from the middle of the shoulders to the head
    V2 = [x2 - x1, 0, z2 - z1]
    # calculate the angle between the two vectors
    angle = math.acos(np.dot(V1,V2) / (np.linalg.norm(V1) * np.linalg.norm(V2)))
    angle = math.degrees(angle)
    if (x2 < x1):
        angle = -angle
    return angle

def angleHeadPitch(head,Rshoulder,Lshoulder):
    # middle of the shoulders
    y1 = (Rshoulder[1] + Lshoulder[1]) / 2
    z1 = (Rshoulder[2] + Lshoulder[2]) / 2
    # coordinates of the head
    y2 = head[1]
    z2 = head[2]
    #vector in -y direction
    V1 = [0,-10,0]
    # vector from the middle of the shoulders to the head
    V2 = [0,y2 - y1, z2 - z1]
    # calculate the angle between the two vectors
    angle = math.acos(np.dot(V1,V2) / (np.linalg.norm(V1) * np.linalg.norm(V2)))
    angle = math.degrees(angle)
    # calibrate the angle
    angle = angle - 60
    return angle
def StandingCrouching(Lhip,Lknee,Lankle,Rhip,Rknee,Rankle):
    # calculate the angle between the leg and the floor
    angle = math.atan((Lhip[2] - Lknee[2]) / (Lhip[1] - Lknee[1]))
    angle = math.degrees(angle)
    if (angle < -60):
        angle = 0
    else:
        angle = 1
    return angle

# class to store angles
class Angles(object):
    def __init__(self):
        self._done = False
        self._body = False
        # here we will store angle data
        self.bodyframe_joint_angle_list = []



# thread to send data
def send_data_thread(sock, angles_detected):
    while not angles_detected._done:
        if angles_detected.bodyframe_joint_angle_list:
            # enviar json a un cliente python 2.7
            bodyframe_landmarks_list_json = pickle.dumps(angles_detected.bodyframe_joint_angle_list, protocol=2)
            sock.sendall(bodyframe_landmarks_list_json)
            time.sleep(1)
            angles_detected.bodyframe_joint_angle_list = []

# Main function
def LandMarksCapture(getcamera):
    angles_detected = Angles()
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_pose = mp.solutions.pose  # Mediapipe Solutions
    cap = cv2.VideoCapture(getcamera)
    font_size = 0.7
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # NOTE: Descomentar para la conexión
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', 10000)
        print('starting up on %s port %s' % server_address)
        sock.connect(server_address)
        print("conectado")
         # # Iniciar el hilo para enviar datos
        send_thread = Thread(target=send_data_thread, args=(sock, angles_detected))
        send_thread.start()

        while True:
            ret, frame = cap.read()
            

            try:
                # Flip the frame horizontally
                # frame = cv2.flip(frame, 1)
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
            
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # get image height and width
                image_height, image_width, _ = image.shape
                 # Make detection
                results = pose.process(image)
                landmarks = results.pose_landmarks.landmark

                # Get coordinates of the right arm
                Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z] 
                Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                
                # Get coordinates of the left arm
                Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z ]
                Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]

                #Get coordinates of the head
                head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y,landmarks[mp_pose.PoseLandmark.NOSE.value].z]
                #these coordinates are to simulate standing up and crouching position
                # get coordinates of the left leg
                Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                # get coordinates of the right leg
                Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]



                # NOTE Calculations for the right arm
                # calculate angle RShoulderPitch

                angleRShoulderPitchDegrees = angleRShoulderPitch(Rshoulder, Relbow)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right Shoulder Pitch: {angleRShoulderPitchDegrees:.2f} degrees",
                            [100,50],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)

                # calculate angle RShoulderRoll
                angleRShoulderRollDegrees = angleRShoulderRoll(Rshoulder, Relbow)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right ShoulderRoll: {angleRShoulderRollDegrees:.2f} degrees",
                            [100,100],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # # calculate angle RElbowRoll
                angleRElbowRollDegrees = angleRElbowRoll(Rshoulder, Relbow, Rwrist)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right ElbowRoll: {angleRElbowRollDegrees:.2f} degrees",
                            [100,150],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # calculate angle RElbowYaw
                angleRElbowYawDegrees = angleRElbowYaw(Relbow,Rwrist, angleRShoulderPitchDegrees)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right ElbowYaw: {angleRElbowYawDegrees:.2f} degrees",
                            [100,200],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                
                # NOTE Calculations for the left arm
                # calculate angle LShoulderPitch
                angleLShoulderPitchDegrees = angleLShoulderPitch(Lshoulder, Lelbow)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left Shoulder Pitch: {angleLShoulderPitchDegrees:.2f} degrees",
                            [image_width-600,50],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # calculate angle LShoulderRoll
                angleLShoulderRollDegrees = angleLShoulderRoll(Lshoulder, Lelbow)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left ShoulderRoll: {angleLShoulderRollDegrees:.2f} degrees",
                            [image_width-600,100],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # # calculate angle LElbowRoll
                angleLElbowRollDegrees = angleLElbowRoll(Lshoulder, Lelbow, Lwrist)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left ElbowRoll: {angleLElbowRollDegrees:.2f} degrees",
                            [image_width-600,150],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # calculate angle LElbowYaw
                angleLElbowYawDegrees = angleLElbowYaw(Lelbow,Lwrist, angleLShoulderPitchDegrees)
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left ElbowYaw: {angleLElbowYawDegrees:.2f} degrees",
                            [image_width-600,200],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # NOTE calculate head angles
                # calculate angle HeadYaw
                angleHeadYawDegrees = angleHeadYaw(head, Rshoulder, Lshoulder)
                # Visualize angle with bigger letters
                cv2.putText(image, f"HeadYaw: {angleHeadYawDegrees:.2f} degrees",
                            [int(image_width/2)-300,50],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                # calculate angle HeadPitch
                angleHeadPitchDegrees = angleHeadPitch(head, Rshoulder, Lshoulder)
                # Visualize angle with bigger letters
                cv2.putText(image, f"HeadPitch: {angleHeadPitchDegrees:.2f} degrees",
                            [int(image_width/2)-300,100],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
                #NOTE calculate standing or crouching if posible
                # calculate standing or crouching
                standingcrouching = StandingCrouching(Lhip,Lknee,Lankle,Rhip,Rknee,Rankle)
                # Visualize angle with bigger letters
                cv2.putText(image, f"StandingCrouching: {standingcrouching}",
                            [int(image_width/2)-300,150],cv2.FONT_HERSHEY_SIMPLEX, font_size, colorlabels, 2, cv2.LINE_AA)
               
                
                angles_detected._body = True
                if angles_detected._body:
                    # adding angles to the list
                    angles_detected.bodyframe_joint_angle_list = [angleRShoulderPitchDegrees, angleRShoulderRollDegrees, angleRElbowRollDegrees, angleRElbowYawDegrees, angleLShoulderPitchDegrees, angleLShoulderRollDegrees, angleLElbowRollDegrees, angleLElbowYawDegrees, angleHeadYawDegrees, angleHeadPitchDegrees]
                    


            except Exception as e:
                # add the line below to see the error
                print(e)

                angles_detected._body = False
                pass
        
        
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
            
            cv2.imshow('Mediapipe Feed', image)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     angles_detected._done = True
            #     break
            # if video ends or q is pressed break
            if cv2.waitKey(10) & 0xFF == ord('q') or ret == False:
                angles_detected._done = True
                break
    
            
        send_thread.join()  # Esperar a que el hilo de envío termine
    cap.release()
    cv2.destroyAllWindows()

                   
# add main function with arguments from terminal
if __name__ == "__main__":
    if len(sys.argv) > 1:
        LandMarksCapture(sys.argv[1])
    else:
        LandMarksCapture(0) 
    # LandMarksCapture(0) # use this line to use the webcam
    # LandMarksCapture("nameofvideo.mp4") # use this line to use a video





