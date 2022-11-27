from cv2 import cv2
from subprocess import call
import mediapipe as mp
import time

webcam = 0
smartphone = 1

circleRadius = 10
fontSize = 1.5
fontThickness = 1

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)

cap = cv2.VideoCapture(webcam)  # 0 for webcam, 1 for external source

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# change volume
call(["osascript -e 'set volume output volume 30'"], shell=True)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip for webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for h_id, lm in enumerate(handLms.landmark):
                # print(h_id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(h_id, cx, cy)
                print(cx, cy)
                if cx - cy == 0:
                    print("ha")

                # wrist
                if h_id == 0:
                    cv2.circle(img, (cx, cy), circleRadius, red, cv2.FILLED)
                    cv2.putText(img, str(int(h_id)), (cx + 15, cy), cv2.FONT_HERSHEY_PLAIN, fontSize, red,
                                fontThickness)
                # thumb tips
                elif h_id == 4:

                    cv2.circle(img, (cx, cy), circleRadius, blue, cv2.FILLED)
                    cv2.putText(img, str(int(h_id)), (cx + 15, cy), cv2.FONT_HERSHEY_PLAIN, fontSize, blue,
                                fontThickness)
                # other tips
                elif h_id in (8, 12, 16, 20):
                    cv2.circle(img, (cx, cy), circleRadius, green, cv2.FILLED)
                    cv2.putText(img, str(int(h_id)), (cx + 15, cy), cv2.FONT_HERSHEY_PLAIN, fontSize, green,
                                fontThickness)
                # other parts
                else:
                    cv2.circle(img, (cx, cy), circleRadius, white, cv2.FILLED)
                    cv2.putText(img, str(int(h_id)), (cx + 15, cy), cv2.FONT_HERSHEY_PLAIN, fontSize, white,
                                fontThickness)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # show fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # show image
    cv2.imshow("test", img)

    cv2.waitKey(1)
