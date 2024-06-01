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
    # Perform object detection
    results = model(frame)

    # Get bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0]

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Function to fetch weather data
def fetch_weather(city):
    api_key = "f2209996d51630a1461af902576269fb"  # Replace with your OpenWeatherMap API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "q=" + city + "&appid=" + api_key + "&units=metric"
    response = requests.get(complete_url)
    return response.json()

# Streamlit app layout
st.title("Welcome to MUNQIDH! - مرحباً بك في منقذ!")

# Sidebar with navigation
st.sidebar.title("Main Menu")
menu_options = ["Home", "Upload", "Overview", "Latest Weather Updates"]
menu_option = st.sidebar.radio("", menu_options)

# Logo placement
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image('/Users/abdullahalharbi/Desktop/last T5/MUNQIDH.png', width=200)  # Adjust width and path to your image
with col3:
    st.image('/Users/abdullahalharbi/Desktop/last T5/998-1.png', width=200)  # Adjust width and path to your image

# Apply CSS styles to highlight the selected menu option in red
st.markdown(f"""
<style>
    .reportview-container {{
        background: white;
        color: black;
    }}
    .sidebar .sidebar-content {{
        background: white;
    }}
    .stButton>button {{
        color: white;
        background-color: #4CAF50;
    }}
    .stText {{
        font-size: 18px;
        font-family: Arial, sans-serif;
    }}
    .css-1d391kg {{
        background-color: {'#ff4b4b' if menu_option == 'Home' else 'white'};
    }}
    .css-1d391kg {{
        background-color: {'#ff4b4b' if menu_option == 'Upload' else 'white'};
    }}
    .css-1d391kg {{
        background-color: {'#ff4b4b' if menu_option == 'Overview' else 'white'};
    }}
    .css-1d391kg {{
        background-color: {'#ff4b4b' if menu_option == 'Latest Weather Updates' else 'white'};
    }}
</style>
""", unsafe_allow_html=True)

if 'running' not in st.session_state:
    st.session_state.running = False

if menu_option == "Home":
    st.subheader("Real-Time Drowning Detection and Safety")
    # Main loop for real-time detection
    frame_placeholder = st.empty()

    if st.session_state.running:
        if st.button("Stop", key="stop_button_home"):
            st.session_state.running = False
    else:
        if st.button("Start", key="start_button_home"):
            st.session_state.running = True

    while st.session_state.running:
        frame = capture_letsview_cast_window()

        if frame is not None:
            # Detect objects in the frame
            frame = detect_objects(frame)
            # Display the frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        time.sleep(0.1)  # Add a small delay to avoid excessive CPU usage

elif menu_option == "Upload":
    st.subheader("Upload a Video for Prediction")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = 1 / fps  # Adjust wait time based on the frame rate

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()

            # Detect objects in the frame
            frame = detect_objects(frame)
            # Display the frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            elapsed_time = time.time() - start_time
            if wait_time > elapsed_time:
                time.sleep(wait_time - elapsed_time)

        cap.release()
        st.write("Video processing completed.")

elif menu_option == "Overview":
    st.subheader("Project Overview")
    st.write("""
        **MUNQIDH** is a real-time drowning detection and safety project using YOLOv8 for object detection.
        The system captures live video streams, detects drowning incidents, and provides alerts to ensure timely intervention.
        Additionally, the project aims to provide weather updates and early warnings for natural disasters.
    """)

elif menu_option == "Latest Weather Updates":
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
