#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import time
import math

from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

from utils import CvFpsCalc

# models
from model import KeyPointClassifier_R
from model import KeyPointClassifier_L
from model import PointHistoryClassifier
from model import MouseClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier_R = KeyPointClassifier_R(invalid_value=8, score_th=0.4)
    keypoint_classifier_L = KeyPointClassifier_L(invalid_value=8, score_th=0.4)
    mouse_classifier = MouseClassifier(invalid_value=2, score_th=0.4)
    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(
            'model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=3)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)
    mouse_id_history = deque(maxlen=40)

    # 靜態手勢最常出現參數初始化
    keypoint_length = 5
    keypoint_R = deque(maxlen=keypoint_length)
    keypoint_L = deque(maxlen=keypoint_length)



    # ========= 使用者自訂姿勢、指令區 =========
    # time.sleep(0.5)
    # keepadd = False

    # ========= 按鍵前置作業 =========
    mode = 0
    presstime = presstime_2 = presstime_3 = presstime_4 = time.time()

    detect_mode = 2
    what_mode = 'mouse'
    landmark_list = 0
    pyautogui.PAUSE = 0

    # ========= 滑鼠前置作業 =========
    wScr, hScr = pyautogui.size()
    frameR = 100
    smoothening = 7
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    mousespeed = 1.5
    clicktime = time.time()

    # ===============================
    i = 0
    finger_gesture_id = 0

    # ========= 主程式運作 =========
    while True:
        left_id = right_id = -1
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) 
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation 
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # print(landmark_list)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
                
                # 靜態手勢資料預測     
                hand_sign_id_R = keypoint_classifier_R(pre_processed_landmark_list)
                hand_sign_id_L = keypoint_classifier_L(pre_processed_landmark_list)              
                mouse_id = mouse_classifier(pre_processed_landmark_list)
                # print(mouse_id)
                if handedness.classification[0].label[0:] == 'Left':
                    left_id = hand_sign_id_L
                
                else:
                    right_id = hand_sign_id_R    

                # 手比one 觸發動態資料抓取
                if right_id == 1 or left_id ==1:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # 動態手勢資料預測
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                # print(finger_gesture_id) # 0 = stop, 1 = clockwise, 2 = counterclockwise, 3 = move,偵測出現的動態手勢

                # 動態手勢最常出現id #########################################
                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                #滑鼠的deque
                mouse_id_history.append(mouse_id)
                most_common_ms_id = Counter(mouse_id_history).most_common()
                # print(f'finger_gesture_history = {finger_gesture_history}')
                # print(f'most_common_fg_id = {most_common_fg_id}')

                # 靜態手勢最常出現id #########################################
                hand_gesture_id = [right_id, left_id]
                keypoint_R.append(hand_gesture_id[0])
                keypoint_L.append(hand_gesture_id[1])
                # print(keypoint_R) # deque右手的靜態id
                # print(most_common_keypoint_id) # 右手靜態id最大
                if right_id != -1:
                    most_common_keypoint_id = Counter(keypoint_R).most_common()
                else:
                    most_common_keypoint_id = Counter(keypoint_L).most_common()

                
                
                # print(f'keypoint = {keypoint}')
                # print(f'most_common_keypoint_id = {most_common_keypoint_id}')

                ###############################################################

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[most_common_keypoint_id[0][0]],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # 偵測是否有手勢 #########################################

        if left_id + right_id > -2:
            if time.time() - presstime > 1:
                # change mode
                if most_common_ms_id[0][0] == 3 and most_common_ms_id[0][1] == 40: #Gesture six changes to the different mode
                    print('Mode has changed')
                    detect_mode = (detect_mode + 1) % 3
                    if detect_mode == 0: what_mode = 'Rest'
                    if detect_mode == 1: what_mode = 'Keyboard'
                    if detect_mode == 2: what_mode = 'Mouse'
                    print(f'Current mode => {what_mode}')
                    presstime = time.time() + 1
                    presstime_4 = time.time()

                # control keyboard
                elif detect_mode == 1:
                    if time.time() - presstime_2 > 1:
                        if time.time() - presstime_4 > 1:
                            # 靜態手勢控制
                            control_keyboard(most_common_keypoint_id, 2, 'K', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 0, 'right', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 7, 'left', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 9, 'C', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 5, 'up', keyboard_TF=True, print_TF=False)
                            control_keyboard(most_common_keypoint_id, 6, 'down', keyboard_TF=True, print_TF=False)
                            presstime_2 = time.time()
                            # print(time.time() - presstime_4 > 1)

                    # 動態手勢控制
                    if most_common_fg_id[0][0] == 1 and most_common_fg_id[0][1] > 12:
                        if time.time() - presstime_3 > 1.5:
                            pyautogui.hotkey('shift', '>')
                            print('speed up')
                            presstime_3 = time.time()
                    elif most_common_fg_id[0][0] == 2 and most_common_fg_id[0][1] > 12:
                        if time.time() - presstime_3 > 1.5:
                            pyautogui.hotkey('shift', '<')
                            print('speed down')
                            presstime_3 = time.time()


                    # print(most_common_fg_id)
                    # print(most_common_fg_id[0][0], most_common_fg_id[0][1])
                    # print(time.time() - presstime_3 > 1)

            if detect_mode == 2:
                if mouse_id == 0:  # Point gesture
                    # print(landmark_list[8]) #index finger
                    # print(landmark_list[12]) #middle finger
                    x1, y1 = landmark_list[8]
                    # x2, y2 = landmark_list[12]
                    # cv.rectangle(debug_image, (frameR, frameR), (cap_width - frameR, cap_height - frameR),
                    #              (255, 0, 255), 2)
                    cv.rectangle(debug_image, (50, 50), (cap_width - 50, cap_height - 100),
                                 (255, 0, 255), 2)
                    # x3 = np.interp(x1, (frameR, cap_width - frameR), (0, wScr))
                    # y3 = np.interp(y1, (frameR, cap_height - frameR), (0, hScr))
                    x3 = np.interp(x1, (50*mousespeed, (cap_width - 50)/mousespeed), (0, wScr))
                    y3 = np.interp(y1, (50*mousespeed, (cap_height - 50)/mousespeed), (0, hScr))
                    # print(x3, y3)
                    # 6. Smoothen Values
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    # 7. Move Mouse
                    pyautogui.moveTo(clocX, clocY)
                    cv.circle(debug_image, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                    plocX, plocY = clocX, clocY

                if mouse_id == 1:
                    length, img, lineInfo = findDistance(landmark_list[8], landmark_list[12], debug_image)

                    # 10. Click mouse if distance short
                    if time.time() - clicktime > 0.5:
                        if length < 40:
                            cv.circle(img, (lineInfo[4], lineInfo[5]),
                                      15, (0, 255, 0), cv.FILLED)
                            pyautogui.click()
                            print('click')
                            clicktime = time.time()


                        # if length > 70:
                        #     cv.circle(img, (lineInfo[4], lineInfo[5]),
                        #               15, (0, 255, 0), cv.FILLED)
                            # pyautogui.click(clicks=2)
                            # print('click*2')
                            # clicktime = time.time()

                if most_common_keypoint_id[0][0] == 5 and most_common_keypoint_id[0][1] == 5:
                    pyautogui.scroll(20)

                if most_common_keypoint_id[0][0] == 6 and most_common_keypoint_id[0][1] == 5:
                    pyautogui.scroll(-20)

                #if left_id == 7 or right_id == 7:
                if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
                    if time.time() - clicktime > 1:
                        pyautogui.click(clicks=2)
                        clicktime = time.time()

                if most_common_keypoint_id[0][0] == 9 and most_common_keypoint_id[0][1] == 5:
                    if time.time() - clicktime > 2:
                        pyautogui.hotkey('alt', 'left')
                        clicktime = time.time()

        cv.putText(debug_image, what_mode, (400, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        # Screen reflection ###################################JL##########################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # 人差指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # 中指
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # 薬指
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # 小指
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # 手の平
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def control_keyboard(most_common_keypoint_id, select_right_id, command, keyboard_TF=True, print_TF=True):
    if most_common_keypoint_id[0][0] == select_right_id and most_common_keypoint_id[0][1] == 5:
        if keyboard_TF:
            pyautogui.press(command)
        if print_TF:
            print(command)
    # print(most_common_keypoint_id)
    # elif select_left_id == -1:
    #     if right_id == -1 and left_id == select_right_id:
    #         if keyboard_TF:
    #             pyautogui.press(command)
    #         if print_TF:
    #             print(command)
    # elif select_right_id == -1:
    #     if left_id == -1 and right_id == select_left_id:
    #         if keyboard_TF:
    #             pyautogui.press(command)
    #         if print_TF:
    #             print(command)


def pick_gesture_command():
    left_number = input('left gesture number :')
    right_number = input('right gesture number :')
    command = input('what command :')
    return int(left_number), int(right_number), command


def pick_number(inputstring):
    keepask = True
    while keepask:
        try:
            number = input(f'{inputstring} :')
            number = int(number)
            if number < -1 or number > 3 or number == 0:
                raise Exception('number is not in range')
        except:
            print('choose again')

        else:
            keepask = False
            # print('choosing nicely')
    return number


def pick_command(inputstring='what command'):
    keepask = True
    while keepask:
        try:
            com = input(f'{inputstring} :')
            com_list = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
                        ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
                        '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
                        'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
                        'browserback', 'browserfavorites', 'browserforward', 'browserhome',
                        'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
                        'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
                        'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
                        'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
                        'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                        'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
                        'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
                        'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
                        'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
                        'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
                        'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
                        'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
                        'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
                        'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
                        'command', 'option', 'optionleft', 'optionright']
            if com not in com_list:
                raise Exception('number is not in range')
        except:
            print('choose again')

        else:
            keepask = False
            print('choosing nicely')
    return com


def findDistance(p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]


if __name__ == '__main__':
    main()