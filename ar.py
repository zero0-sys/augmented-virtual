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
GRAVITY = 1.1
FLOOR_Y = 380
camera_offset_x = 0

# =========================
# OBJECTS & ANCHORS
# =========================
anchors = []
objects = []

# DEFAULT OBJECT (AUTO MUNCUL)
anchors.append({"x": 320, "y": 150})
objects.append({
    "anchor": anchors[0],
    "local_x": 0,
    "local_y": 0,
    "size": 70,
    "vy": 0,
    "grabbed": False
})

anchor_cooldown = 0

# =========================
# UTILS
# =========================
def hand_centroid(lm, w, h):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return int(sum(xs)/len(xs)*w), int(sum(ys)/len(ys)*h)

def palm_center(lm, w, h):
    x = int((lm[0].x + lm[9].x) / 2 * w)
    y = int((lm[0].y + lm[9].y) / 2 * h)
    return x, y

def is_open_palm(lm):
    return (
        lm[8].y < lm[6].y and
        lm[12].y < lm[10].y and
        lm[16].y < lm[14].y
    )

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    right_hand_pos = None
    left_hand_open = False

    if res.multi_hand_landmarks:
        for i, hand in enumerate(res.multi_hand_landmarks):
            lm = hand.landmark
            label = res.multi_handedness[i].classification[0].label

            cx, cy = hand_centroid(lm, w, h)
            px, py = palm_center(lm, w, h)

            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

            if label == "Right":
                right_hand_pos = (px, py)

            if label == "Left" and is_open_palm(lm):
                left_hand_open = True

    # =========================
    # CREATE NEW ANCHOR
    # =========================
    if left_hand_open and right_hand_pos and anchor_cooldown == 0:
        anchors.append({
            "x": right_hand_pos[0] + camera_offset_x,
            "y": right_hand_pos[1]
        })

        objects.append({
            "anchor": anchors[-1],
            "local_x": -35,
            "local_y": -35,
            "size": 70,
            "vy": 0,
            "grabbed": False
        })

        anchor_cooldown = 40

    if anchor_cooldown > 0:
        anchor_cooldown -= 1

    # =========================
    # OBJECT INTERACTION (PALM GRAB)
    # =========================
    for obj in objects:
        world_x = obj["anchor"]["x"] + obj["local_x"]
        world_y = obj["anchor"]["y"] + obj["local_y"]
        screen_x = world_x - camera_offset_x
        screen_y = world_y

        obj_center = (
            screen_x + obj["size"]//2,
            screen_y + obj["size"]//2
        )

        if right_hand_pos:
            hx, hy = right_hand_pos
            dist_to_obj = math.hypot(hx - obj_center[0], hy - obj_center[1])

            if dist_to_obj < 90:
                obj["grabbed"] = True
                obj["vy"] = 0
                obj["local_x"] = hx + camera_offset_x - obj["anchor"]["x"] - obj["size"]/2
                obj["local_y"] = hy - obj["anchor"]["y"] - obj["size"]/2
            else:
                obj["grabbed"] = False
        else:
            obj["grabbed"] = False

    # =========================
    # PHYSICS
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
    # DRAW ANCHORS
    # =========================
    for a in anchors:
        ax = int(a["x"] - camera_offset_x)
        ay = int(a["y"])
        cv2.circle(frame, (ax, ay), 6, (255,0,0), -1)

    # =========================
    # DRAW OBJECTS
    # =========================
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
        "Palm Grab AR | Right: Move | Left: New Anchor",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.imshow("AR Anchor Palm Interaction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
