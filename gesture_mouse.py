import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

click_cooldown = 0.3  # seconds between clicks
last_click_time = 0

def fingers_up(lm_list):
    fingers = []
    # Thumb: compare tip (4) with IP joint (3) horizontally (x)
    fingers.append(lm_list[4][1] > lm_list[3][1])  # True if thumb up (right hand)
    # Fingers: tip y < pip y means finger is up (for index, middle, ring, pinky)
    fingers.append(lm_list[8][2] < lm_list[6][2])   # Index
    fingers.append(lm_list[12][2] < lm_list[10][2]) # Middle
    fingers.append(lm_list[16][2] < lm_list[14][2]) # Ring
    fingers.append(lm_list[20][2] < lm_list[18][2]) # Pinky
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(lm_list)
            all_fingers_closed = not any(fingers)

            # Move pointer using index finger tip (landmark 8)
            index_finger = lm_list[8]
            x = int(index_finger[1] * screen_width / w)
            y = int(index_finger[2] * screen_height / h)
            pyautogui.moveTo(x, y)

            # Pinch gesture detection (thumb tip 4 and index tip 8)
            thumb_finger = lm_list[4]
            distance = ((index_finger[1] - thumb_finger[1]) ** 2 + (index_finger[2] - thumb_finger[2]) ** 2) ** 0.5
            if distance < 40:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

            # Click when index and middle fingers are up together, and NOT all fingers closed
            current_time = time.time()
            if fingers[1] and fingers[2] and not all_fingers_closed:
                if (current_time - last_click_time) > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
