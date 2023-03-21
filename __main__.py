#!/usr/bin/env python3

import sys

import cv2
import mediapipe as mp

from lib.logger import Logger
from lib.motion import UniversalHandMotionTracker

class Vision:
    def __init__(self):
        self.logger = Logger(filename="debug.log")
        self.uhmt = UniversalHandMotionTracker()

    def bootstrap(self):
        # Initialize hand tracking
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, # false for video o/w batch img frames
                               max_num_hands=1, 
                               model_complexity=1, #0/1 increase latency and accuracy
                               min_detection_confidence=0.5, # 0-1 """
                               min_tracking_confidence=0.5) # """ 
        #hands = self.mp_hands.Hands(static_image_mode=False,
        #                       max_num_hands=1, min_detection_confidence=0.7)
        # Initialize OpenCV camera
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # Ignoring empty camera frame
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for tip, lm in enumerate(hand_landmarks.landmark):
                        # keep track of all fingers for every fram
                        # if for 30 frames it moves 30% of screen size
                        # do slide and ignore next 30 frames
                        # repeat
                        # cmds: 1 hand left,right,up,down slides
                        # close hand to select
                        # 2 hand slide bar
                        h, w, c = image.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        self.logger.debug(f"{tip}, {cx}, {cy}")
                        cache = self.uhmt.parseGesture(tip, cx, cy)
                        self.logger.debug(cache)
                    mp_drawing.draw_landmarks(image,
                                              hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        return 0

if __name__ == '__main__':
    vision = Vision()
    ret = vision.bootstrap()
    sys.exit(ret)
