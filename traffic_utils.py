# import cv2
# from ultralytics import YOLO
# import pandas as pd
# import time

# model = YOLO("yolov8n.pt")  # Lightweight and fast

# def process_live_feed(duration=30):
#     cap = cv2.VideoCapture(0)  # 0 = default webcam
#     start_time = time.time()
#     log_data = []

#     while True:
#         ret, frame = cap.read()
#         if not ret or (time.time() - start_time > duration):
#             break

#         results = model.predict(source=frame, device='cpu')[0]
#         timestamp = round(time.time() - start_time, 2)

#         car_count = sum(1 for box in results.boxes.cls if int(box) == 2)
#         bus_count = sum(1 for box in results.boxes.cls if int(box) == 5)
#         truck_count = sum(1 for box in results.boxes.cls if int(box) == 7)
#         person_count = sum(1 for box in results.boxes.cls if int(box) == 0)

#         log_data.append({
#             "timestamp_sec": timestamp,
#             "car_count": car_count,
#             "bus_count": bus_count,
#             "truck_count": truck_count,
#             "person_count": person_count
#         })

#     cap.release()
#     df = pd.DataFrame(log_data)
#     return df


# traffic_utils.py

# import cv2
# from ultralytics import YOLO
# import pandas as pd
# import time

# model = YOLO("yolov8n.pt")  # Pretrained model

# def process_live_feed(duration=30):
#     cap = cv2.VideoCapture(0)  # Default webcam
#     start_time = time.time()
#     log_data = []

#     while True:
#         ret, frame = cap.read()
#         if not ret or (time.time() - start_time > duration):
#             break

#         results = model.predict(source=frame, device='cpu')[0]
#         timestamp = round(time.time() - start_time, 2)

#         car_count = sum(1 for box in results.boxes.cls if int(box) == 2)
#         bus_count = sum(1 for box in results.boxes.cls if int(box) == 5)
#         truck_count = sum(1 for box in results.boxes.cls if int(box) == 7)
#         person_count = sum(1 for box in results.boxes.cls if int(box) == 0)

#         # Draw boxes on frame
#         annotated_frame = results.plot()

#         # Show the frame in a separate OpenCV window
#         cv2.imshow("Live Feed - Press Q to stop early", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         log_data.append({
#             "timestamp_sec": timestamp,
#             "car_count": car_count,
#             "bus_count": bus_count,
#             "truck_count": truck_count,
#             "person_count": person_count
#         })

#     cap.release()
#     cv2.destroyAllWindows()
#     df = pd.DataFrame(log_data)
#     return df


# traffic_utils.py

# import cv2
# from ultralytics import YOLO
# import pandas as pd
# import time
# import numpy as np
# from PIL import Image

# model = YOLO("yolov8n.pt")  # Pretrained lightweight YOLO model

# def process_live_feed_streamlit(duration=10, frame_display=None):
#     cap = cv2.VideoCapture(0)
#     start_time = time.time()
#     log_data = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret or (time.time() - start_time > duration):
#             break

#         # Run YOLO inference
#         results = model.predict(source=frame, device='cpu')[0]

#         # Draw annotated boxes
#         annotated_frame = results.plot()

#         # Count objects
#         timestamp = round(time.time() - start_time, 2)
#         car_count = sum(1 for box in results.boxes.cls if int(box) == 2)
#         bus_count = sum(1 for box in results.boxes.cls if int(box) == 5)
#         truck_count = sum(1 for box in results.boxes.cls if int(box) == 7)
#         person_count = sum(1 for box in results.boxes.cls if int(box) == 0)

#         log_data.append({
#             "timestamp_sec": timestamp,
#             "car_count": car_count,
#             "bus_count": bus_count,
#             "truck_count": truck_count,
#             "person_count": person_count
#         })

#         # Convert to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(frame_rgb)

#         # Show in Streamlit using placeholder
#         if frame_display:
#             frame_display.image(pil_img, caption=f"Timestamp: {timestamp}s", use_column_width=True)

#     cap.release()
#     df = pd.DataFrame(log_data)
#     return df


import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

def process_frame_for_streamlit(frame):
    results = model.predict(source=frame, device='cpu')[0]

    car_count = sum(1 for cls in results.boxes.cls if int(cls) == 2)
    bus_count = sum(1 for cls in results.boxes.cls if int(cls) == 5)
    truck_count = sum(1 for cls in results.boxes.cls if int(cls) == 7)
    person_count = sum(1 for cls in results.boxes.cls if int(cls) == 0)

    annotated_frame = results.plot()
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    return pil_image, {
        "car_count": car_count,
        "bus_count": bus_count,
        "truck_count": truck_count,
        "person_count": person_count
    }
