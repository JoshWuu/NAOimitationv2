



import socket
import mediapipe as mp
import cv2
import time
import pickle
from threading import Thread

class landmark(object):
    def __init__(self):
        self._done = False
        # here we will store skeleton data
        self._body = None
        # Every body frame joint list variable (format vector4D[jointRef, joint.x, joint.y, joint.z])
        self.bodyframe_joint_list = []

    def complete_bodyframe_joint_list(self, LM):
        for i in range(0, 32):
            self.bodyframe_joint_list.append(self.landmark_to_joint3D(LM, i))

    def landmark_to_joint3D(self, LM, i):
        x = LM.landmark[i].x
        y = LM.landmark[i].y
        z = LM.landmark[i].z
        joint3D = [i, x, y, z]
        return joint3D


def send_data_thread(sock, body_detected):
    while not body_detected._done:
        if body_detected.bodyframe_joint_list:
            # enviar json a un cliente python 2.7
            bodyframe_landmarks_list_json = pickle.dumps(body_detected.bodyframe_joint_list, protocol=2)
            sock.sendall(bodyframe_landmarks_list_json)
            time.sleep(1)
            print(body_detected.bodyframe_joint_list)
            print(len(body_detected.bodyframe_joint_list))
            body_detected.bodyframe_joint_list = []


def LandMarksCapture():
    body_detected = landmark()
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    # Drawing specs
    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    # cap
    cap = cv2.VideoCapture(0)

    # initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # NOTE: Descomentar para la conexión
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', 10000)
        print('starting up on %s port %s' % server_address)
        sock.connect(server_address)
        print("conectado")

        # Iniciar el hilo para enviar datos
        send_thread = Thread(target=send_data_thread, args=(sock, body_detected))
        send_thread.start()

        while cap.isOpened():
            ret, frame = cap.read()
            # frame = cv2.flip(frame,1)
            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make detections
            results = holistic.process(image)
            # NOTE Revisar si hay deteccion de cuerpos
            if results.pose_landmarks:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw body pose connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2), )
                # Draw left hand connections
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # Draw right hand connections
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                cv2.imshow('Holistic Model detection', image)
                # Guardar el cuerpo detectado
                body_detected._body = results.pose_landmarks
            else:
                print("no body detected")
                body_detected._body = None

            if body_detected._body is not None:
                Landmarksread = results.pose_landmarks
                body_detected.complete_bodyframe_joint_list(Landmarksread)
                # extract the coordinates of the landmarks

            if cv2.waitKey(10) & 0xFF == ord('q'):
                body_detected._done = True
                break

        cap.release()
        cv2.destroyAllWindows()
        send_thread.join()  # Esperar a que el hilo de envío termine


if __name__ == '__main__':
    LandMarksCapture()
    

