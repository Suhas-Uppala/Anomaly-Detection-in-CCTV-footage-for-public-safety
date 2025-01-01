import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client
import geocoder
from tempfile import NamedTemporaryFile
import os
from datetime import datetime


model = load_model('Suspicious_Human_Activity_Detection_LRCN_Model.h5')

# Twilio configuration
account_sid = 'AC4e7eb78b90d176256ebfd74b912314ff'
auth_token = '7666260ba3947ddf32458015571359b0'
twilio_number = '+16095282776'
recipient_number = '+917989665270'

client = Client(account_sid, auth_token)

def get_location():
    g = geocoder.ip('me')
    location = g.latlng
    if location:
        place = geocoder.osm(location, method='reverse')
        if place and place.address:
            return place.address
        else:
            return f"Location identified with coordinates (latitude, longitude): {location}. Address details are currently unavailable."
    else:
        return "Location not available"

def send_alert():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = get_location()
    message_body = f"Alert: Suspicious activity has been detected in the monitored area. Location: {location}. Time of detection: {timestamp}. Please review immediately."
    message = client.messages.create(
        body=message_body,
        from_=twilio_number,
        to=recipient_number
    )
    st.write(f"Alert successfully sent to +917989XXXXX with detected location: {location}.")

def preprocess_frame(frame):
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

st.markdown(
    """
    <style>
    .title {
        color: #FF5733; /* Choose your desired color */
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="title">VisionGuard: Detecting Anomalies, Securing Lives</p>', unsafe_allow_html=True)
st.title('VisionGuard: Detecting Anomalies, Securing Lives')

# Initialize session state
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

if 'crime_detected' not in st.session_state:
    st.session_state.crime_detected = False

if 'video_filename' not in st.session_state:
    st.session_state.video_filename = None

if 'out' not in st.session_state:
    st.session_state.out = None

def clear_history():
    st.session_state.processing_done = False
    st.session_state.crime_detected = False
    if st.session_state.temp_file_path:
        if os.path.exists(st.session_state.temp_file_path):
            os.remove(st.session_state.temp_file_path)
        st.session_state.temp_file_path = None
    if st.session_state.video_filename:
        if os.path.exists(st.session_state.video_filename):
            os.remove(st.session_state.video_filename)
        st.session_state.video_filename = None
    if st.session_state.out:
        st.session_state.out.release()
        st.session_state.out = None
    st.write("History cleared.")

uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])

if uploaded_file and not st.session_state.processing_done:

    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        st.session_state.temp_file_path = temp_file.name

    cap = cv2.VideoCapture(st.session_state.temp_file_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    stframe = st.empty()

    crime_counter = 0
    alert_sent = False

    
    #st.session_state.video_filename = 'normal_detection.avi'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)

        crime_prob = prediction[0][0]
        weapon_prob = prediction[0][1]

        label = 'Normal Activity'
        color = (0, 255, 0)

        if crime_prob > 0.5:
            st.session_state.crime_detected = True
            crime_counter += 1
            if crime_counter == 5 and not alert_sent:
                send_alert()
                alert_sent = True

            if weapon_prob > 0.5:
                label = 'Crime & Weapon Detected'
            else:
                label = 'Crime Detected'
            color = (0, 0, 255)

        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

       
        if st.session_state.out is None:
            
            if st.session_state.crime_detected:
                st.session_state.video_filename = 'crime_detected.avi'
            else:
                st.session_state.video_filename = 'normal_detection.avi'
            
            st.session_state.out = cv2.VideoWriter(st.session_state.video_filename, fourcc, fps, (frame_width, frame_height))

        st.session_state.out.write(frame)

    cap.release()
    if st.session_state.out:
        st.session_state.out.release()
    st.write(f"Processing complete.")
    st.session_state.processing_done = True

if st.session_state.processing_done:
    if st.button('Clear History'):
        clear_history()