"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

from cv2 import cv2
import mediapipe as mp
import time
import math

circleRadius = 10
fontSize = 1.5
fontThickness = 2  # integer only

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
white = (255, 255, 255)
purple = (255, 0, 255)


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for h_id, lm in enumerate(myHand.landmark):
                # print(h_id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(h_id, cx, cy)
                self.lmList.append([h_id, cx, cy])
                if draw:
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

        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), circleRadius - 5, purple, cv2.FILLED)
            cv2.circle(img, (x2, y2), circleRadius - 5, purple, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), purple, 3)
            cv2.circle(img, (cx, cy), circleRadius - 5, purple, cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    tipIds = [4, 8, 12, 16, 20]
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # flip for webcam
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            fingers = []

            # thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 other tips
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            print(fingers)
            totalFingers = fingers.count(1)
            print(totalFingers)

            # 엄지(4번)과 검지(8번)이 만나는 동작
            if detector.findDistance(4, 8, img)[0] <= 40:
                print("엄지+검지")
            # 엄지(4번)과 중지(12번)이 만나는 동작
            elif detector.findDistance(4, 12, img)[0] <= 40:
                print("엄지+중지")
            # 엄지(4번)과 약지(16번)이 만나는 동작
            elif detector.findDistance(4, 16, img)[0] <= 40:
                print("엄지+약지")
            # 엄지(4번)과 소지(20번)이 만나는 동작
            elif detector.findDistance(4, 20, img)[0] <= 40:
                print("엄지+새끼")
            # 엄지(4번)으로 검지 둘째 마디(5번과 6번 사이)를 누르는 동작

            # 가위(브이자) 동작
            # 바위(주먹 쥐기) 동작
            # 보(손바닥 펴기) 동작

            # 뻐큐 동작
            # 피쓰 동작
            # 권총(화살표) 동작
            # 문고리 돌리기 동작
            # 숫자(0,1,2,3,4,5,6,7,8,9) 동작

            # else:
                # print("0")

        # show fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # show result
        cv2.imshow("HandTrackingModule", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
