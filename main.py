import cv2
import mediapipe as mp
import numpy as np
import torch
print("pakage imported....")

cam = cv2.VideoCapture(0)
wedith = cam.set(3,700) # 3 is id for wedith
height = cam.set(4,480) # 4 is id for height

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame , 1)
    imgrgp = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgp)

    inscreenlist = []
    if result.multi_hand_landmarks:
        for handInScreen in result.multi_hand_landmarks:
            for id , inscreen in enumerate(handInScreen.landmark):
                h,w,c = frame.shape
                cx, cy = int(inscreen.x * w ), int(inscreen.y * h)
                inscreenlist.append([id,cx,cy])
                print(inscreenlist)
                mpdraw.draw_landmarks(frame ,handInScreen ,mpHands.HAND_CONNECTIONS)
        print("hand detect")

    cv2.imshow("handtracker" ,frame)
    if cv2.waitKey(1) & 0xff == 27:
        break
cam.release()
cv2.destroyAllWindows()
