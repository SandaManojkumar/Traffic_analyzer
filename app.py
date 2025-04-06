# # app.py

# import streamlit as st
# from traffic_utils import process_live_feed_streamlit
# import matplotlib.pyplot as plt

# st.title("🚦 Real-time Traffic Analyzer")

# option = st.selectbox("Choose Analysis Mode", ["Upload Video", "Real-Time Webcam"])

# if option == "Real-Time Webcam":
#     if st.button("Start Real-Time Analysis"):
#         stframe = st.empty()  # Placeholder for video stream

#         with st.spinner("Analyzing live traffic..."):
#             df = process_live_feed_streamlit(duration=10, frame_display=stframe)

#         st.success("✅ Live analysis completed!")
#         st.write(df)

#         # Plot counts over time
#         st.line_chart(df[['car_count', 'bus_count', 'truck_count', 'person_count']])


# import streamlit as st
# from traffic_utils import model
# import cv2
# import pandas as pd
# import time
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# # Set up the Streamlit app layout
# st.set_page_config(layout="wide")
# st.title("🚦 Real-time Traffic Analyzer with Live Graph")

# # Set duration with a slider (up to 5 minutes)
# duration = st.slider("Duration (seconds)", 10, 300, 60, step=10)

# # Start button
# if st.button("Start Real-Time Analysis"):
#     stframe = st.empty()  # For video frame
#     chart_placeholder = st.empty()  # For live graph

#     cap = cv2.VideoCapture(0)
#     start_time = time.time()
#     log_data = []

#     while True:
#         ret, frame = cap.read()
#         if not ret or (time.time() - start_time > duration):
#             break

#         # Run YOLO prediction
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

#         # Annotate and display frame
#         annotated = results.plot()
#         annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
#         stframe.image(Image.fromarray(annotated_rgb), caption=f"Time: {timestamp}s", use_container_width=True)

#         # Live graph update
#         df = pd.DataFrame(log_data)
#         chart_placeholder.line_chart(df[['car_count', 'bus_count', 'truck_count', 'person_count']])

#     cap.release()

#     st.success("✅ Live analysis completed!")
#     st.write("📊 Final Report")
#     st.write(df)

#     # Download CSV
#     csv = df.to_csv(index=False).encode()
#     st.download_button("Download CSV Report", csv, "traffic_report.csv", "text/csv")


# import streamlit as st
# from traffic_utils import process_frame_for_streamlit
# import cv2
# import pandas as pd
# import time
# from PIL import Image
# import tempfile

# st.set_page_config(layout="wide")
# st.title("🚦 Real-time Traffic Analyzer with Live Graph")

# mode = st.selectbox("Choose Input Mode", ["Real-Time Webcam", "Upload Video"])
# duration = st.slider("Duration (seconds, webcam only)", 10, 300, 60)

# stframe = st.empty()
# chart_placeholder = st.empty()

# log_data = []

# def process_video(cap, duration=None):
#     start_time = time.time()
#     while True:
#         ret, frame = cap.read()
#         if not ret or (duration and (time.time() - start_time > duration)):
#             break

#         frame = cv2.resize(frame, (640, 360))
#         timestamp = round(time.time() - start_time, 2)

#         processed_image, counts = process_frame_for_streamlit(frame)
#         counts["timestamp_sec"] = timestamp
#         log_data.append(counts)

#         stframe.image(processed_image, caption=f"⏱ Time: {timestamp}s", use_container_width=True)
#         df = pd.DataFrame(log_data)
#         chart_placeholder.line_chart(df[["car_count", "bus_count", "truck_count", "person_count"]])
#     cap.release()

# if st.button("Start Analysis"):
#     if mode == "Real-Time Webcam":
#         cap = cv2.VideoCapture(0)
#         process_video(cap, duration=duration)

#     elif mode == "Upload Video":
#         uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])
#         if uploaded_file is not None:
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(uploaded_file.read())
#             cap = cv2.VideoCapture(tfile.name)
#             process_video(cap)

#     if log_data:
#         st.success("✅ Analysis completed!")
#         df = pd.DataFrame(log_data)
#         st.write("📊 Final Report")
#         st.write(df)

#         csv = df.to_csv(index=False).encode()
#         st.download_button("Download CSV Report", csv, "traffic_report.csv", "text/csv")

import streamlit as st
from traffic_utils import process_frame_for_streamlit
import cv2
import pandas as pd
import time
from PIL import Image
import tempfile

st.set_page_config(layout="wide")
st.title("🚦 Real-time Traffic Analyzer with Live Graph")

mode = st.radio("Choose Analysis Mode", ["Real-Time Webcam", "Upload Video File"])
duration = st.slider("Duration for Live Analysis (in seconds)", 10, 300, 60)

stframe = st.empty()
chart_placeholder = st.empty()

log_data = []

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    log_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(time.time() - start_time, 2)
        annotated_image, counts = process_frame_for_streamlit(frame)

        log_data.append({"timestamp_sec": timestamp, **counts})

        stframe.image(annotated_image, caption=f"Time: {timestamp}s", use_container_width=True)
        df = pd.DataFrame(log_data)
        chart_placeholder.line_chart(df[['car_count', 'bus_count', 'truck_count', 'person_count']])

    cap.release()
    return pd.DataFrame(log_data)

if mode == "Real-Time Webcam":
    if st.button("Start Real-Time Analysis"):
        cap = cv2.VideoCapture(0)
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (time.time() - start_time > duration):
                break

            timestamp = round(time.time() - start_time, 2)
            annotated_image, counts = process_frame_for_streamlit(frame)

            log_data.append({"timestamp_sec": timestamp, **counts})
            stframe.image(annotated_image, caption=f"Time: {timestamp}s", use_container_width=True)

            df = pd.DataFrame(log_data)
            chart_placeholder.line_chart(df[['car_count', 'bus_count', 'truck_count', 'person_count']])

        cap.release()

        st.success("✅ Live analysis completed!")
        st.write("📊 Final Report")
        st.write(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV Report", csv, "traffic_report.csv", "text/csv")

elif mode == "Upload Video File":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        with st.spinner("Processing uploaded video..."):
            df = analyze_video(temp_video_path)

        st.success("✅ Video analysis completed!")
        st.write("📊 Final Report")
        st.write(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV Report", csv, "traffic_report.csv", "text/csv")

