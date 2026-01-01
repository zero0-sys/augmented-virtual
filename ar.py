import cv2
import mediapipe as mp
import numpy as np
import math

# =========================
# CAMERA SOURCE
# =========================
IP_WEBCAM_URL = "http://192.168.1.5:8080/video"
cap = cv2.VideoCapture(IP_WEBCAM_URL, cv2.CAP_FFMPEG)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# WORLD SETTINGS
# =========================
GRAVITY = 1.2
FLOOR_Y = 380
camera_offset_x = 0

# =========================
# ANCHORS & OBJECTS
# =========================
anchors = []
objects = []

# 🔥 DEFAULT ANCHOR + OBJECT (BIAR KELIHATAN LANGSUNG)
anchors.append({"x": 320, "y": 150})
objects.append({
    "anchor": anchors[0],
    "local_x": 0,
    "local_y": 0,
    "size": 60,
    "vy": 0,
    "grabbed": False
})

anchor_cooldown = 0

# =========================
# UTILS
# =========================
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def is_open_palm(lm):
    return lm[8].y < lm[6].y and lm[12].y < lm[10].y

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🔥 RESIZE BIAR NGGAK GEDE
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    pinch = False
    pinch_point = None
    left_hand_open = False

    if res.multi_hand_landmarks:
        for i, hand in enumerate(res.multi_hand_landmarks):
            lm = hand.landmark
            label = res.multi_handedness[i].classification[0].label

            thumb = (int(lm[4].x*w), int(lm[4].y*h))
            index = (int(lm[8].x*w), int(lm[8].y*h))

            # RIGHT HAND → DRAG
            if label == "Right":
                if dist(thumb, index) < 40:
                    pinch = True
                    pinch_point = (
                        (thumb[0] + index[0]) // 2,
                        (thumb[1] + index[1]) // 2
                    )
                    cv2.circle(frame, pinch_point, 8, (0,255,0), -1)

            # LEFT HAND → CREATE ANCHOR
            if label == "Left" and is_open_palm(lm):
                left_hand_open = True

    # =========================
    # CREATE NEW ANCHOR (ANTI SPAM)
    # =========================
    if left_hand_open and pinch_point and anchor_cooldown == 0:
        anchors.append({
            "x": pinch_point[0] + camera_offset_x,
            "y": pinch_point[1]
        })

        objects.append({
            "anchor": anchors[-1],
            "local_x": -30,
            "local_y": -30,
            "size": 60,
            "vy": 0,
            "grabbed": False
        })

        anchor_cooldown = 30  # ~1 detik

    if anchor_cooldown > 0:
        anchor_cooldown -= 1

    # =========================
    # OBJECT INTERACTION
    # =========================
    for obj in objects:
        world_x = obj["anchor"]["x"] + obj["local_x"]
        world_y = obj["anchor"]["y"] + obj["local_y"]
        screen_x = world_x - camera_offset_x
        screen_y = world_y

        if pinch and pinch_point:
            if (screen_x < pinch_point[0] < screen_x + obj["size"] and
                screen_y < pinch_point[1] < screen_y + obj["size"]):

                obj["grabbed"] = True
                obj["vy"] = 0
                obj["local_x"] = pinch_point[0] + camera_offset_x - obj["anchor"]["x"]
                obj["local_y"] = pinch_point[1] - obj["anchor"]["y"]
        else:
            obj["grabbed"] = False

    # =========================
    # PHYSICS (GRAVITY)
    # =========================
    for obj in objects:
        if not obj["grabbed"]:
            obj["vy"] += GRAVITY
            obj["local_y"] += obj["vy"]

            if obj["anchor"]["y"] + obj["local_y"] + obj["size"] > FLOOR_Y:
                obj["local_y"] = FLOOR_Y - obj["anchor"]["y"] - obj["size"]
                obj["vy"] = 0

    # =========================
    # DRAW FLOOR
    # =========================
    cv2.line(frame, (0, FLOOR_Y), (640, FLOOR_Y), (255,255,255), 2)

    # =========================
    # DRAW ANCHORS & OBJECTS
    # =========================
    for a in anchors:
        ax = int(a["x"] - camera_offset_x)
        ay = int(a["y"])
        cv2.circle(frame, (ax, ay), 5, (255,0,0), -1)

    for obj in objects:
        x = int(obj["anchor"]["x"] + obj["local_x"] - camera_offset_x)
        y = int(obj["anchor"]["y"] + obj["local_y"])
        cv2.rectangle(
            frame,
            (x, y),
            (x + obj["size"], y + obj["size"]),
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "Right: Pinch=Drag | Left: Open Palm=New Anchor",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow("AR Anchor Multi-Hand", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
