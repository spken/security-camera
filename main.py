import cv2
import numpy as np
import winsound
import threading
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

roi = [(100, 100), (500, 100), (500, 400), (100, 400)]

with open("coco.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=2, fy=1.3)
    if not ret:
        break

    results = model(frame)

    detections = results[0].boxes
    boxes = detections.xywh.cpu().numpy()
    confidences = detections.conf.cpu().numpy()
    class_ids = detections.cls.cpu().numpy().astype(int)
    
    cv2.polylines(frame, [np.array(roi, np.int32)], True, (255, 0, 0), 2)

    for i, (x, y, w, h) in enumerate(boxes):
        x1, y1, x2, y2 = (
            int((x - w / 2)),
            int((y - h / 2)),
            int((x + w / 2)),
            int((y + h / 2)),
        )

        if classes[class_ids[i]] == "person" and cv2.pointPolygonTest(np.array(roi, np.int32), (x, y), False) >= 0:
            cv2.putText(frame, "INTRUDER!", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            threading.Thread(target=winsound.Beep(1000, 500), daemon=True).start()  # TODO: supa laggy

        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv2.imshow("Security Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
