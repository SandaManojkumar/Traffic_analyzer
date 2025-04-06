# traffic_processor.py

import cv2
from ultralytics import YOLO
import pandas as pd
import json
import time

def process_video(video_path, output_csv_path, output_json_path):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    log_data = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(time.time() - start_time, 2)
        results = model.predict(source=frame, device='cpu')[0]

        car_count = bus_count = truck_count = person_count = 0

        for box in results.boxes.data.tolist():
            _, _, _, _, _, cls = box
            cls = int(cls)

            if cls == 0:
                person_count += 1
            elif cls == 2:
                car_count += 1
            elif cls == 5:
                bus_count += 1
            elif cls == 7:
                truck_count += 1

        log_data.append({
            "timestamp_sec": timestamp,
            "car_count": car_count,
            "bus_count": bus_count,
            "truck_count": truck_count,
            "person_count": person_count
        })

    cap.release()

    df = pd.DataFrame(log_data)
    df.to_csv(output_csv_path, index=False)

    with open(output_json_path, 'w') as f:
        json.dump(log_data, f, indent=4)

    return df
