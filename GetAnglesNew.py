#program to see the angles of the robot on the screen and send them to the robot
# it can detect camera as well as a video
import cv2
import mediapipe as mp
import numpy as np
import sys
def angle_between_vectors(v1, v2, normal_vector):
    v1 = np.array(v1)
    v2 = np.array(v2)
    normal_vector = np.array(normal_vector)

    # Resto del código sigue igual...
    proj_v1 = v1 - np.dot(v1, normal_vector) / np.linalg.norm(normal_vector)**2 * normal_vector
    proj_v2 = v2 - np.dot(v2, normal_vector) / np.linalg.norm(normal_vector)**2 * normal_vector
    cos_theta = np.dot(proj_v1, proj_v2) / (np.linalg.norm(proj_v1) * np.linalg.norm(proj_v2))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def angleRShoulderPitch(Lshoulder,Rhip,Rshoulder,Relbow): #calulates the Shoulderpitch value for the Left shoulder by using geometry
    # get the vector from the shoulder to the elbow
    upper_arm = [Relbow[0] - Rshoulder[0], Relbow[1] - Rshoulder[1], Relbow[2] - Rshoulder[2]]
    # get the vector from the shoulder to the hip
    shoulder_hip = [Rhip[0] - Rshoulder[0], Rhip[1] - Rshoulder[1], Rhip[2] - Rshoulder[2]]
    #get the vector from the left shoulder to the right shoulder
    shoulder_shoulder = [Lshoulder[0] - Rshoulder[0], Lshoulder[1] - Rshoulder[1], Lshoulder[2] - Rshoulder[2]]
    # calculate the angle between the vectors
    angle = angle_between_vectors(upper_arm, shoulder_hip, shoulder_shoulder)
    return angle


def LandMarksCapture(video):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # cap = cv2.VideoCapture("arms.mov")
    cap = cv2.VideoCapture(video)

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
        while True:
            ret, frame = cap.read()
            # Flip the frame horizontally
            # frame = cv2.flip(frame, 1)
            # Recolor image to RGB
            
            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = image.shape
                # Extract landmarks

                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates of the left arm
                Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z ]
                Lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]


                # Get coordinates of the right arm
                Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z] 
                Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

                # Get coordinates of the hip
                RHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]


                # calculate angle RShoulderPitch
                angleRShoulderPitchDegrees = angleRShoulderPitch(Lshoulder,RHip,Rshoulder, Relbow)


                # NOTE Calculations for the right arm
                # calculate angle RShoulderPitch

                angleRShoulderPitchDegrees = 100
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right Shoulder Pitch: {angleRShoulderPitchDegrees:.2f} deg",
                            [100,50],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)

                # calculate angle RShoulderRoll
                angleRShoulderRollDegrees = 200
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right ShoulderRoll: {angleRShoulderRollDegrees:.2f} deg",
                            [100,100],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # # calculate angle RElbowRoll
                angleRElbowRollDegrees = 300
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right ElbowRoll: {angleRElbowRollDegrees:.2f} deg",
                            [100,150],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # calculate angle RElbowYaw
                angleRElbowYawDegrees = 400
                # Visualize angle with bigger letters
                cv2.putText(image, f"Right ElbowYaw: {angleRElbowYawDegrees:.2f} deg",
                            [100,200],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                
                # NOTE Calculations for the left arm
                # calculate angle LShoulderPitch
                angleLShoulderPitchDegrees = 500
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left Shoulder Pitch: {angleLShoulderPitchDegrees:.2f} deg",
                            [image_width-700,50],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # calculate angle LShoulderRoll
                angleLShoulderRollDegrees =600
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left ShoulderRoll: {angleLShoulderRollDegrees:.2f} deg",
                            [image_width-700,100],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # # calculate angle LElbowRoll
                angleLElbowRollDegrees = 700
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left ElbowRoll: {angleLElbowRollDegrees:.2f} deg",
                            [image_width-700,150],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # calculate angle LElbowYaw
                angleLElbowYawDegrees = 800
                # Visualize angle with bigger letters
                cv2.putText(image, f"Left ElbowYaw: {angleLElbowYawDegrees:.2f} deg",
                            [image_width-700,200],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # NOTE calculate head angles
                # calculate angle HeadYaw
                angleHeadYawDegrees = 900
                # Visualize angle with bigger letters
                cv2.putText(image, f"HeadYaw: {angleHeadYawDegrees:.2f} deg",
                            [int(image_width/2)-300,50],cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorlabels, 2, cv2.LINE_AA)
                # calculate angle HeadPitch
                angleHeadPitchDegrees = 1000
                # Visualize angle with bigger letters
                cv2.putText(image, f"HeadPitch: {angleHeadPitchDegrees:.2f} deg",
                            [int(image_width/2)-300,100],cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorlabels, 2, cv2.LINE_AA)
                #NOTE calculate standing or crouching if posible
                # calculate standing or crouching
                standingcrouching =1100
                # Visualize angle with bigger letters
                cv2.putText(image, f"StandingCrouching: {standingcrouching}",
                            [int(image_width/2)-300,150],cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorlabels, 2, cv2.LINE_AA)
                print(image_width, image_height)    
            except Exception as e:
                print(e)
                pass
            #delay for 0.5 seconds
            # time.sleep(0.5)
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or ret == False:
                break

        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    if len(sys.argv) > 1:
        LandMarksCapture(sys.argv[1])
    else:
        LandMarksCapture(0)
    # LandMarksCapture(0) # use this line to use the webcam
    # LandMarksCapture("nameofvideo.mp4") # use this line to use a video
