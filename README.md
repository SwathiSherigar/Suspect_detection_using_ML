
# Suspect Image Retrieval using Face Recognition through CCTV Footage

## Overview

This project is designed to detect and recognize faces from CCTV footage using face recognition techniques. The system matches faces from uploaded images against faces in the video footage, helping identify and retrieve potential suspects.

## Features

- **Image Encoding**: Upload multiple images of suspects to encode their facial features.
- **Video Processing**: Upload CCTV footage for face detection and recognition.
- **Face Matching**: Compares faces in the footage with the uploaded images and labels matches with the suspect's name.

## Requirements

To run this project, ensure you have the following dependencies installed:

```bash
pip install streamlit opencv-python-headless mtcnn  numpy Pillow
```

## How to Run

1. Clone the repository to your local machine.
2. Install the required libraries using the `pip` command above.
3. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

4. Upload images of suspects and a CCTV video in the web interface to start the face recognition process.



## How It Works

- **Face Detection**: Uses the MTCNN algorithm to detect faces in both images and video frames.
- **Face Encoding**: Encodes facial features of uploaded images for matching.
- **Face Matching**: Compares face encodings from the video footage with suspect image encodings and highlights matches on the video.

## Usage Instructions

1. **Upload Suspect Images**: Upload one or more images of suspects that you want to search for in the CCTV footage.
2. **Upload CCTV Footage**: Upload a video (MP4, AVI, or MOV format) for face recognition.
3. **Start Processing**: Click the "Start Video Processing" button to begin detecting and matching faces in the footage.
4. **Results**: Matched faces will be highlighted in the video frames.
