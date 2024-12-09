import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import cv2
import numpy as np
import tempfile
import os

# Initialize the object detection pipeline
object_detector = pipeline("object-detection",
                         model="facebook/detr-resnet-50")

def draw_bounding_boxes(frame, detections):
    """
    Draws bounding boxes on the video frame based on the detections.
    """
    # Convert numpy array to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Use default font
    font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        xmin = int(box['xmin'])
        ymin = int(box['ymin'])
        xmax = int(box['xmax'])
        ymax = int(box['ymax'])

        # Draw the bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        # Create label with score
        label = detection['label']
        score = detection['score']
        text = f"{label} {score:.2f}"

        # Draw text with background rectangle for visibility
        text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle([
            (text_bbox[0], text_bbox[1]),
            (text_bbox[2], text_bbox[3])
        ], fill="red")
        draw.text((xmin, ymin), text, fill="white", font=font)

    # Convert back to numpy array
    frame_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_with_boxes

def create_output_writer(cap, output_path):
    """
    Create video writer with different codecs, trying multiple options
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Try different codecs
    codecs = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    for codec, ext in codecs:
        try:
            output_file = os.path.splitext(output_path)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
            
            if out is not None and out.isOpened():
                return out, output_file
                
        except Exception as e:
            print(f"Failed with codec {codec}: {str(e)}")
            continue
            
    raise ValueError("Could not initialize any video codec")

def frame_to_pil(frame):
    """Convert OpenCV frame to PIL Image"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

def process_video(video_path, progress=gr.Progress()):
    """
    Process the video file and return the path to the processed video
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.expanduser("~"), "Videos", "ObjectDetection")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output path
        output_path = os.path.join(output_dir, "output_video.mp4")
        
        # Initialize video writer
        out, output_path = create_output_writer(cap, output_path)

        frame_count = 0
        process_every_n_frames = 1  # Process every frame
        
        progress(0, desc="Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Process frame
            if frame_count % process_every_n_frames == 0:
                # Convert frame to PIL Image for the model
                pil_frame = frame_to_pil(frame)
                
                try:
                    # Detect objects
                    detections = object_detector(pil_frame)
                    
                    # Draw bounding boxes
                    frame = draw_bounding_boxes(frame, detections)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    # Continue with the original frame if detection fails
                    pass
            
            # Write the frame
            out.write(frame)
            
            # Update progress
            progress((frame_count / total_frames), desc=f"Processing frame {frame_count}/{total_frames}")

        # Release everything
        cap.release()
        out.release()
        
        # Verify the output file exists and has size
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("Output video file is empty or was not created")
            
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise gr.Error(f"Error processing video: {str(e)}")

def detect_objects_in_video(video):
    """
    Gradio interface function for video object detection
    """
    if video is None:
        raise gr.Error("Please upload a video file")
    
    try:
        # Process the video
        output_path = process_video(video)
        return output_path
        
    except Exception as e:
        raise gr.Error(f"Error during video processing: {str(e)}")

# Create the Gradio interface
demo = gr.Interface(
    fn=detect_objects_in_video,
    inputs=[
        gr.Video(label="Upload Video")
    ],
    outputs=[
        gr.Video(label="Processed Video")
    ],
    title="@GenAILearniverse Project: Video Object Detection",
    description="""
    Upload a video to detect and track objects within it. 
    The application will process the video and draw bounding boxes around detected objects 
    with their labels and confidence scores.
    Note: Processing may take some time depending on the video length.
    """,
    examples=[],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()