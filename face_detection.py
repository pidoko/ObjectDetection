import cv2
import argparse
import gradio as gr
import numpy as np
import logging

# Configure logging for production-level traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FACE_CASCADE_PATH: str = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Initialize the Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_classifier.empty():
    raise RuntimeError(f"Error loading Haar cascade classifier from {FACE_CASCADE_PATH}")

def detect_faces(frame: np.ndarray) -> np.ndarray:
    """
    Detect faces in the given image frame and draw bounding boxes around them.
    
    The image is converted to grayscale for detection using Haar cascades, and 
    then the bounding boxes are drawn on the original color frame.
    
    Args:
        frame (np.ndarray): Input image in BGR color space.
        
    Returns:
        np.ndarray: The image with detected face bounding boxes drawn.
    """
    # Convert the input frame to grayscale as required by the face detector
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with specified parameters (scaleFactor, minNeighbors, and minSize)
    faces = face_classifier.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(40, 40)
    )
    
    # Draw bounding boxes around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    return frame

def process_video(frame: np.ndarray) -> np.ndarray:
    """
    Process a single video frame for face detection, designed for use with Gradio.

    Converts the frame from RGB (as provided by Gradio) to BGR for OpenCV processing, 
    applies face detection, and then converts the frame back to RGB.

    Args:
        frame (np.ndarray): Input video frame in RGB color space.

    Returns:
        np.ndarray: Processed video frame with face detection annotations in RGB color space.
                    If the input frame is empty or None, returns the frame unchanged.
    """
    # Check if the frame is empty
    if frame is None or frame.size == 0:
        logging.warning("Received an empty frame from Gradio. Skipping processing.")
        return frame

    try:
        # Convert from RGB to BGR for OpenCV compatibility
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except cv2.error as e:
        logging.error("Error converting frame from RGB to BGR: %s", e)
        return frame

    processed_frame = detect_faces(bgr_frame)

    try:
        # Convert the processed frame back to RGB for Gradio
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logging.error("Error converting processed frame from BGR to RGB: %s", e)
        return processed_frame

    return rgb_frame

def launch_gradio(source: int = 0) -> None:
    """
    Launch a Gradio web interface for real-time face detection.
    
    This function sets up the Gradio interface using the `process_video` function as
    the processing endpoint. The interface is configured to stream video from the specified source.
    
    Args:
        source (int): Video source index (default is 0 for the primary webcam).
    """
    iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(source, streaming=True),
        outputs=gr.Image(type="numpy"),
        live=True,
        title="Real-Time Face Detection",
        description="This application detects faces in real-time using Haar cascades."
    )
    
    # Launch the Gradio interface with a public sharing link enabled
    iface.launch(share=True)

def main(video_source: int = 0) -> None:
    """
    Capture video from a specified source and perform real-time face detection.
    
    This function uses OpenCV to capture video frames, applies face detection on each frame,
    and displays the annotated frames. The video stream can be terminated by pressing 's'.
    
    Args:
        video_source (int): Video source index (default is 0 for the primary webcam).
    """
    video_capture = cv2.VideoCapture(video_source)

    if not video_capture.isOpened():
        logging.error("Error: Could not open video source %s", video_source)
        return

    logging.info("Starting video stream. Press 's' to exit.")

    while True:
        success, frame = video_capture.read()

        # Handle empty frames by continuing the loop instead of breaking
        if not success or frame is None or frame.size == 0:
            logging.warning("Received an empty frame. Waiting for valid frames...")
            continue  # Keep waiting for valid frames
        
        # Process the frame to detect and annotate faces
        processed_frame = detect_faces(frame)
        
        # Display the processed frame in a window
        cv2.imshow('Face Detector', processed_frame)

        # Exit the loop when 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            logging.info("Exit command received. Closing video stream.")
            break

    # Release resources and close windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face detection in a video stream using Haar cascades."
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Video source index (default: 0 for the primary webcam)"
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Run face detection as a Gradio web app"
    )
    
    args = parser.parse_args()

    if args.gradio:
        launch_gradio(source=args.source)
    else:
        main(video_source=args.source)
