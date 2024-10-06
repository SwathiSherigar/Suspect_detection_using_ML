import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
import face_recognition
import os
import tempfile
from PIL import Image

# Initialize the MTCNN face detector
detector = MTCNN()

# Create a temporary directory for uploaded files
temp_dir = tempfile.mkdtemp()

# Function to load and encode images
def load_and_encode_images(image_paths):
    encodings = []
    names = []
    
    for image_path in image_paths:
        # Load and convert the image to RGB
        input_image = cv2.imread(image_path)
        rgb_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        faces = detector.detect_faces(rgb_input_image)
        
        # If no face detected, skip this image
        if len(faces) == 0:
            st.warning(f"No face detected in the image: {image_path}")
            continue
        
        # Extract face locations and encodings
        face_locations = [(face['box'][1], face['box'][0] + face['box'][2], face['box'][1] + face['box'][3], face['box'][0]) for face in faces]
        input_face_encodings = face_recognition.face_encodings(rgb_input_image, face_locations)
        
        # Save the first encoding (assuming one face per image)
        encodings.append(input_face_encodings[0])
        names.append(os.path.basename(image_path).split(".")[0])
    
    return encodings, names

# Function to process video and search for faces
def search_faces_in_video(video_path, known_face_encodings, known_face_names):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        faces_in_frame = detector.detect_faces(rgb_frame)
        if len(faces_in_frame) == 0:
            continue  # Skip if no faces were found

        # Extract face locations
        frame_face_locations = [(face['box'][1], face['box'][0] + face['box'][2], face['box'][1] + face['box'][3], face['box'][0]) for face in faces_in_frame]
        frame_face_encodings = face_recognition.face_encodings(rgb_frame, frame_face_locations)

        for (top, right, bottom, left), face_encoding in zip(frame_face_locations, frame_face_encodings):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                continue  # Skip if no known faces are available
            
            best_match_index = np.argmin(face_distances)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if matches[best_match_index]:
            # Convert frame to a format Streamlit can display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, channels="RGB", use_column_width=True)

    video_capture.release()

# Streamlit application
st.title("🔍 Real-time Face Detection and Recognition")

# Set custom background color and image
st.markdown("""
    <style>
    .stApp {
        background-image:url("man.png");
        background-size: cover;
        background-position: center;
    }
    </style>
""", unsafe_allow_html=True)

st.write("Upload images for face encoding and a video for face recognition.")

# Image upload
uploaded_images = st.file_uploader("Choose images...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
image_paths = []

if uploaded_images:
    for img_file in uploaded_images:
        image_path = os.path.join(temp_dir, img_file.name)
        with open(image_path, "wb") as f:
            f.write(img_file.getbuffer())
        image_paths.append(image_path)

    # Step 1: Load and encode the input images
    known_face_encodings, known_face_names = load_and_encode_images(image_paths)
    st.success("🟢 Images loaded and encoded successfully.")

# Video upload
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Process video on user request
if uploaded_video and known_face_encodings and known_face_names:
    video_path = os.path.join(temp_dir, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Display a button to start video processing
    process_video = st.button("Start Video Processing")

    if process_video:
        st.info("⚙ Processing video...")
        search_faces_in_video(video_path, known_face_encodings, known_face_names)
        st.success("✅ Video processing completed!")