# Video Object Detection using Gradio
# Overview
This project provides an interactive web application for detecting and tracking objects in videos. It uses the DEtection Transformer (DETR) model from Hugging Face to identify and label objects in the video. The app processes each frame of the video, detects objects, and draws bounding boxes with the object labels and confidence scores. The processed video is then returned to the user for download.

# Features
Real-Time Object Detection: Detects and tracks multiple objects in video frames.
Bounding Boxes: Draws bounding boxes around detected objects, displaying labels and confidence scores.
Video Processing: Processes video frames in sequence and outputs the processed video with visual enhancements.
User-Friendly Interface: Built with Gradio to easily upload and download videos.
Multiple Codec Support: Uses various codecs for video processing to ensure compatibility across platforms.
# How It Works
Input Video:

Users upload a video file they want to analyze.
Object Detection:

The app uses the DETR (Detection Transformer) model to detect and label objects in each video frame.
Output Video:

The application processes each frame of the video, draws bounding boxes around detected objects, and outputs the processed video with labeled bounding boxes.
Download the Video:

After processing, the app provides a link to download the output video with tracked objects.
# Installation
To run this project locally, you will need Python and some dependencies. Follow the instructions below to set up the project.

Prerequisites
Python 3.7 or later
Pip (Python package installer)
GPU (Optional but recommended for faster processing)
Steps to Install Locally
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/video-object-detection-gradio.git
cd video-object-detection-gradio
Install Dependencies: Create a virtual environment (optional but recommended) and install the required dependencies:

bash
Copy code
pip install -r requirements.txt
This will install:

gradio for creating the web interface.
transformers for the DETR object detection pipeline.
opencv-python for video processing.
PIL (Python Imaging Library) for drawing bounding boxes.
Run the Application:

bash
Copy code
python app.py
Access the App: After running the application, open your browser and go to: http://localhost:7860 Here, you can upload a video and start the object detection process.

# Usage
Upload a Video:
Click on the video input area and select the video file you want to process. Supported formats include .mp4, .avi, .mov, etc.
Processing the Video:
Once the video is uploaded, the application will start processing the video and detecting objects.
A progress bar will show the processing status.
Download the Processed Video:
After the video has been processed, the app will generate a downloadable version with bounding boxes drawn around the detected objects.
You will be able to download the video directly from the interface.
# Example
Input Video:
A 10-second clip showing a car driving on a busy street.

Output Video:
A processed video where each frame has bounding boxes drawn around detected objects like the car, pedestrians, and traffic signs, along with labels and confidence scores.

Code Explanation
# Key Components
Object Detection Pipeline:

The app uses the facebook/detr-resnet-50 model from Hugging Face to detect objects. The pipeline function is used to load and interact with the model.
Drawing Bounding Boxes:

The draw_bounding_boxes function takes each frame, detects objects, and draws bounding boxes with the corresponding labels and confidence scores.
Video Processing:

The process_video function reads each frame of the uploaded video, processes it through the object detection pipeline, and writes the processed frames to a new video file.
Video Writer:

The create_output_writer function tries different codecs to write the processed video to disk in various formats like .mp4 or .avi.
Gradio Interface:

The Gradio interface allows users to interact with the app, upload videos, and download the processed videos.
# Dependencies
This project relies on the following Python libraries:

gradio: For creating the web interface.
transformers: For using the pretrained object detection model (DETR).
opencv-python: For reading, processing, and writing video frames.
Pillow (PIL): For handling image manipulation and drawing bounding boxes.
To install these dependencies, run:

bash
Copy code
pip install -r requirements.txt
# File Structure
php
Copy code
video-object-detection-gradio/
│
├── app.py                # Main script to run the Gradio interface
├── requirements.txt      # Python dependencies
├── README.md             # Documentation for the project
└── example_images/       # Folder for example images (optional)
# Limitations
Performance: Processing large or long videos might take time, depending on the system's hardware, especially if no GPU is available.
File Size Limit: Gradio has some size restrictions for file uploads. If the video is too large, consider compressing it before uploading.
# Future Enhancements
Add support for multi-object tracking across frames.
Allow users to specify the objects to detect from a predefined set.
Support video segmentation and analysis in addition to object detection.
# Contributing
We welcome contributions! If you'd like to improve this project:

Fork the repository.
Create a feature branch.
Commit your changes and submit a pull request.
# License
This project is licensed under the MIT License. See the LICENSE file for more details.
