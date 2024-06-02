import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from ultralytics import YOLO
import streamlit as st
import requests
import tempfile
import streamlit.components.v1 as components
import time
from collections import defaultdict
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Welcome to MUNQIDH! - مرحباً بك في منقذ!", page_icon="🌊")

# Load YOLOv8 model
model_path = '/Users/abdullahalharbi/Desktop/last T5/best (1).pt'
model = YOLO(model_path)

# Define function to capture LetsView [Cast] window
def capture_letsview_cast_window():
    # Find the LetsView [Cast] window
    windows = gw.getWindowsWithTitle('LetsView [Cast]')
    if not windows:
        st.write("LetsView [Cast] window not found.")
        return None

    window = windows[0]  # Take the first window with the title

    # Get the window position and size
    left, top, width, height = window.left, window.top, window.width, window.height

    # Capture the specific region
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

# Define function to detect objects
def detect_objects(frame):
    # Resize the frame for faster processing
    original_size = frame.shape[1], frame.shape[0]
    resized_frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    results = model(resized_frame)

    detections = []
    labels = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0]

            # Scale the bounding box back to the original frame size
            x1, y1, x2, y2 = int(x1 * original_size[0] / 640), int(y1 * original_size[1] / 480), int(x2 * original_size[0] / 640), int(y2 * original_size[1] / 480)

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append((x1, y1, x2, y2))
            labels.append(label)
    return frame, detections, labels

# Function to fetch weather data
def fetch_weather(city):
    api_key = "f2209996d51630a1461af902576269fb"  # Replace with your OpenWeatherMap API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city + "&appid=" + api_key + "&units=metric"
    response = requests.get(complete_url)
    return response.json()

# Page functions
def home_page():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image('/Users/abdullahalharbi/Desktop/last T5/MUNQIDH.png', width=180)  # Adjust width and path to your image
    with col3:
        st.image('/Users/abdullahalharbi/Desktop/last T5/998-1.png', width=160)  # Adjust width and path to your image

    st.subheader("Real-Time Detecting Stream...")
    frame_placeholder = st.empty()
    table_placeholder = st.empty()

    trackers = {}
    unique_counts = defaultdict(int)
    counted_ids = set()
    next_id = 0
    frame_counter = 0
    reset_interval = 30  # Number of frames after which the tracking info is reset

    frame = capture_letsview_cast_window()

    while frame is not None:
        frame_counter += 1
        if frame_counter % reset_interval == 0:
            trackers.clear()
            counted_ids.clear()

        # Detect objects in the frame
        if frame_counter % 3 == 0:  # Process every 3rd frame
            frame, detections, labels = detect_objects(frame)

            for i, (x1, y1, x2, y2) in enumerate(detections):
                new_detection = True
                for obj_id, (tracker, _) in trackers.items():
                    success, bbox = tracker.update(frame)
                    if success:
                        tx1, ty1, tw, th = map(int, bbox)
                        tx2, ty2 = tx1 + tw, ty1 + th
                        if abs(x1 - tx1) < 20 and abs(y1 - ty1) < 20 and abs(x2 - tx2) < 20 and abs(y2 - ty2) < 20:
                            new_detection = False
                            break

                if new_detection:
                    tracker = initialize_tracker()
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, bbox)
                    trackers[next_id] = (tracker, labels[i])
                    if next_id not in counted_ids:
                        unique_counts[labels[i]] += 1
                        counted_ids.add(next_id)
                    next_id += 1

        # Update trackers
        to_remove = []
        for obj_id, (tracker, label) in trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                to_remove.append(obj_id)

        # Remove failed trackers
        for obj_id in to_remove:
            del trackers[obj_id]

        # Display the frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        # Display count table
        table_placeholder.table(pd.DataFrame.from_dict(unique_counts, orient='index', columns=['Count']))

    st.write("""
        **What is MUNQIDH?**

        **MUNQIDH** is an innovative search and rescue system that utilizes drones, specifically the DJI Mini 2,
            to detect vehicles and people submerged in floods. The project aims to enhance rescue operations by
            providing real-time video feeds, automating detection processes, and facilitating rapid response.
    """)

def upload_page():
    st.subheader("Upload a Video for Prediction")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()
        table_placeholder = st.empty()
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = 1 / fps  # Adjust wait time based on the frame rate

        trackers = {}
        unique_counts = defaultdict(int)
        counted_ids = set()
        next_id = 0
        frame_counter = 0
        reset_interval = 30  # Number of frames after which the tracking info is reset

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()

            frame_counter += 1
            if frame_counter % reset_interval == 0:
                trackers.clear()
                counted_ids.clear()

            # Detect objects in the frame
            if frame_counter % 3 == 0:  # Process every 3rd frame
                frame, detections, labels = detect_objects(frame)

                for i, (x1, y1, x2, y2) in enumerate(detections):
                    new_detection = True
                    for obj_id, (tracker, _) in trackers.items():
                        success, bbox = tracker.update(frame)
                        if success:
                            tx1, ty1, tw, th = map(int, bbox)
                            tx2, ty2 = tx1 + tw, ty1 + th
                            if abs(x1 - tx1) < 20 and abs(y1 - ty1) < 20 and abs(x2 - tx2) < 20 and abs(y2 - ty2) < 20:
                                new_detection = False
                                break

                    if new_detection:
                        tracker = initialize_tracker()
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        tracker.init(frame, bbox)
                        trackers[next_id] = (tracker, labels[i])
                        if next_id not in counted_ids:
                            unique_counts[labels[i]] += 1
                            counted_ids.add(next_id)
                        next_id += 1

            # Update trackers
            to_remove = []
            for obj_id, (tracker, label) in trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    x1, y1, w, h = map(int, bbox)
                    x2, y2 = x1 + w, y1 + h
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    to_remove.append(obj_id)

            # Remove failed trackers
            for obj_id in to_remove:
                del trackers[obj_id]

            # Display the frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            # Display count table
            table_placeholder.table(pd.DataFrame.from_dict(unique_counts, orient='index', columns=['Count']))

            elapsed_time = time.time() - start_time
            if wait_time > elapsed_time:
                time.sleep(wait_time - elapsed_time)

        cap.release()
        st.write("Video processing completed.")

def weather_updates_page():
    st.subheader("Latest Weather Updates")
    cities = ["Riyadh", "Jeddah", "Mecca", "Medina", "Dammam"]
    selected_city = st.selectbox("Select a city to get the current weather:", cities)

    if st.button("Get Weather", key="get_weather_button"):
        weather_data = fetch_weather(selected_city)
        if weather_data.get("cod") != "404":
            main = weather_data.get("main", {})
            weather_desc = weather_data.get("weather", [{}])[0].get("description", "No description available")
            st.write(f"Temperature: {main.get('temp', 'N/A')}°C")
            st.write(f"Humidity: {main.get('humidity', 'N/A')}%")
            st.write(f"Weather Description: {weather_desc}")
        else:
            st.write("City Not Found.")

    # Embed Windy radar
    st.subheader("Weather Radar")
    components.iframe("https://embed.windy.com/embed2.html?lat=24.7136&lon=46.6753&zoom=5&level=surface&overlay=rain&product=ecmwf&menu=&message=true&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=&detailLat=24.7136&detailLon=46.6753&metricWind=default&metricTemp=default&radarRange=-1", height=600)

# Sidebar navigation
def main():
    with st.sidebar:
        st.title("MUNQIDH 🌊 منقذ")

        if st.button("Home"):
            st.session_state.page = "Home"
        if st.button("Upload"):
            st.session_state.page = "Upload"
        if st.button("Latest Weather Updates"):
            st.session_state.page = "Latest Weather Updates"

    page = st.session_state.get("page", "Home")

    if page == "Home":
        home_page()
    elif page == "Upload":
        upload_page()
    elif page == "Latest Weather Updates":
        weather_updates_page()

if __name__ == "__main__":
    main()
