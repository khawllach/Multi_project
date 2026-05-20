import cv2

cap = cv2.VideoCapture("video.mov")

frames = []

for i in range(20):
    ret, frame = cap.read()
    if not ret:
        break

    if i == 0:
        cv2.imwrite("frame1.png", frame)

    if i == 5:
        cv2.imwrite("frame2.png", frame)

cap.release()