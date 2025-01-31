import cv2
import argparse
import gradio as gr
import numpy as np

# Load the Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_classifier.empty():
    raise RuntimeError("Error loading Haar cascade classifier")

def detect_faces(frame):
    """Detects faces in the given frame and draws bounding boxes."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return faces

def process_video(frame):
    """Processes incoming video frames for face detection in Gradio."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
    frame = detect_faces(frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display

def launch_gradio(source=0):
    """Launches the Gradio interface for face detection."""
    iface = gr.Interface(
        fn=process_video, 
        inputs=gr.Video(source, streaming=True),
        outputs=gr.Image(type="numpy"),
        live=True
    )
    iface.launch(share=True) # Enables public link

def main(video_source=0):
    """Main function to start video capture and face detection."""
    video_capture = cv2.VideoCapture(video_source)

    if not video_capture.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to exit the program")

    while True:
        success, frame = video_capture.read() 
        if not success:
            print(f"Error: Failed to read frame.\n")
            break

        detect_faces(frame)
        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection in video stream.")
    parser.add_argument("--source", type=int, default=0, help="Video source (default: 0 for webcam)")
    parser.add_argument("--gradio", action="store_true", help="Run face detection as a Gradio web app")
    
    args = parser.parse_args()

    if args.gradio:
        launch_gradio(0)
    else:
        main(args.source)