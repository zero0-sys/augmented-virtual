import cv2
import mediapipe as mp

# =========================
# CAMERA
# =========================
IP_WEBCAM_URL = "http://192.168.1.5:8080/video"
cap = cv2.VideoCapture(IP_WEBCAM_URL, cv2.CAP_FFMPEG)

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# LANDMARK NAME MAP
# =========================
finger_names = {
    0: "wrist",

    1: "thumb 1", 2: "thumb 2", 3: "thumb 3", 4: "thumb 4",
    5: "index 1", 6: "index 2", 7: "index 3", 8: "index 4",
    9: "middle 1",10: "middle 2",11: "middle 3",12: "middle 4",
    13: "ring 1",14: "ring 2",15: "ring 3",16: "ring 4",
    17: "pinky 1",18: "pinky 2",19: "pinky 3",20: "pinky 4"
}

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
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = result.multi_handedness[i].classification[0].label

            # Warna beda kiri / kanan
            if label == "Left":
                point_color = (0, 0, 255)   # merah
            else:
                point_color = (255, 0, 0)   # biru

            # ===== DRAW CONNECTIONS (TULANG) =====
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2),
                mp_draw.DrawingSpec(color=point_color, thickness=2)
            )

            # ===== DRAW & LABEL EACH POINT =====
            for idx, lm in enumerate(hand_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 4, point_color, -1)

                # tampilkan nama landmark
                name = finger_names.get(idx, "")
                cv2.putText(
                    frame,
                    name,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,255),
                    1
                )

            # Label tangan kiri / kanan
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(
                frame,
                label,
                (cx - 20, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    cv2.imshow("Hand Landmark Debug", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
