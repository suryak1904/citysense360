import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO


class CCTVService:
    """
    CCTV Surveillance Service
    Supports image and video analysis using YOLOv8
    """

    PERSON_CLASS = 0
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        output_dir: str = "outputs/cctv",
        image_output_dir: str = "outputs/cctv_images",
        event_log: str = "outputs/cctv_events.json",
    ):
        self.model = YOLO(model_path)

        self.output_dir = output_dir
        self.image_output_dir = image_output_dir
        self.event_log = event_log

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)


    def _estimate_speed(self, prev_center, curr_center, time_diff):
        if prev_center is None or time_diff == 0:
            return 0
        dist = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
        return dist / time_diff

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def _log_event(self, event: dict):
        with open(self.event_log, "a") as f:
            f.write(json.dumps(event) + "\n")



    def process_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")

        results = self.model(img, verbose=False)[0]
        boxes = results.boxes

        people_count = 0
        vehicle_count = 0

        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == self.PERSON_CLASS:
                people_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if cls in self.VEHICLE_CLASSES:
                vehicle_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        alerts = []
        if people_count > 15:
            alerts.append("CROWD")
            cv2.putText(img, "CROWD DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(img, f"People: {people_count}", (20, img.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Vehicles: {vehicle_count}", (20, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        output_path = os.path.join(
            self.image_output_dir,
            f"annotated_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, img)

        event = {
            "timestamp": datetime.now().isoformat(),
            "source": "CCTV_IMAGE",
            "people_count": people_count,
            "vehicle_count": vehicle_count,
            "alerts": alerts
        }

        self._log_event(event)

        return img, event, output_path


    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).replace(".mp4", "")
        output_video_path = os.path.join(
            self.output_dir,
            f"{video_name}_annotated.mp4"
        )

        prev_centers = {}
        stationary_time = {}
        prev_time = time.time()

        out = None
        people_count = 0
        vehicle_count = 0
        accident_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if out is None:
                out = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (frame.shape[1], frame.shape[0])
                )

            results = self.model(frame, verbose=False)[0]
            boxes = results.boxes

            people_count = 0
            vehicle_count = 0
            vehicle_boxes = []

            curr_time = time.time()
            time_diff = curr_time - prev_time
            prev_time = curr_time

            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cls == self.PERSON_CLASS:
                    people_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cls in self.VEHICLE_CLASSES:
                    vehicle_count += 1
                    vehicle_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    speed = self._estimate_speed(
                        prev_centers.get((cx, cy)), (cx, cy), time_diff
                    )

                    if speed > 30:
                        cv2.putText(frame, "OVERSPEED", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if speed < 2:
                        stationary_time[(cx, cy)] = stationary_time.get((cx, cy), 0) + time_diff
                        if stationary_time[(cx, cy)] > 5:
                            cv2.putText(frame, "ILLEGAL PARKING", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        stationary_time[(cx, cy)] = 0

                    prev_centers[(cx, cy)] = (cx, cy)

            for i in range(len(vehicle_boxes)):
                for j in range(i + 1, len(vehicle_boxes)):
                    if self._iou(vehicle_boxes[i], vehicle_boxes[j]) > 0.3:
                        accident_detected = True
                        cv2.putText(frame, "POSSIBLE ACCIDENT", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if people_count > 15:
                cv2.putText(frame, "CROWD DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, f"People: {people_count}", (20, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        if out:
            out.release()

        alerts = []
        if people_count > 15:
            alerts.append("CROWD")
        if accident_detected:
            alerts.append("ACCIDENT")

        event = {
            "timestamp": datetime.now().isoformat(),
            "source": os.path.basename(video_path),
            "people_count": people_count,
            "vehicle_count": vehicle_count,
            "alerts": alerts
        }

        self._log_event(event)

        return output_video_path, event



if __name__ == "__main__":
    service = CCTVService()
    img, event, path = service.process_image("sample.jpg")
    print(event)
