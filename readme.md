# Real-Time Face Detection Application

A production-ready Python application for real-time face detection using Haar cascades. This project supports both local video stream processing via OpenCV and web-based streaming using Gradio.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Gradio Web App](#gradio-web-app)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

This project implements a robust real-time face detection system that leverages OpenCV's Haar cascade classifiers. The application supports:
- Direct processing from a video capture device (e.g., webcam)
- A web-based interface using Gradio for remote access and demonstration purposes

The design emphasizes production-level code practices including detailed documentation, modular design, error handling, and logging for enhanced traceability.

## Features

- **Real-Time Face Detection:** Processes video frames in real time to detect and annotate faces.
- **Gradio Integration:** Provides a web-based interface for live streaming and remote access.
- **Configurable Video Source:** Easily specify the video input source.
- **Robust Error Handling:** Utilizes Python's logging framework for better debugging and production monitoring.
- **Modular Code:** Organized functions with detailed docstrings and type annotations for maintainability.

## Prerequisites

- Python 3.7 or higher
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [Gradio](https://gradio.app/)
- [NumPy](https://numpy.org/)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/face-detection-app.git
   cd face-detection-app

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

## Usage
The application supports two main usage modes: a local video stream for face detection using OpenCV, and a Gradio-based web         interface for remote access


- **Command-Line Interface**
To run the application using a local video source (e.g., a webcam), execute:

python face_detector.py --source 0
--source: Specifies the video source index (default is 0 for the primary webcam).

- **Gradio Web App**
To launch the application as a Gradio web app, run:
python face_detector.py --gradio --source 0
--gradio: Launches the Gradio interface.
--source: Specifies the video source index (default is 0 for the primary webcam).
Upon launching, Gradio provides a public link (if sharing is enabled) which can be accessed from any device.

- **Configuration**
Cascade Classifier:
The Haar cascade XML file is loaded from OpenCV’s default data directory. If needed, update the FACE_CASCADE_PATH constant in the code to point to a custom classifier file.

Logging:
Logging is configured at the INFO level. Modify the logging configuration in the source code for different verbosity or to integrate with external logging systems.

Testing
While unit tests are not included by default, it is recommended to write tests for critical components, especially the face detection logic. Consider using frameworks such as pytest for writing and executing tests.

Contributing
Contributions are welcome! If you have improvements, bug fixes, or new features to add, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes with clear messages.
Push to your fork and open a pull request.
Please ensure that your code adheres to the existing style and that you update documentation as necessary.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
OpenCV for providing the Haar cascade implementation.
Gradio for the easy-to-use web interface.
The open-source community for valuable libraries and tools.

## Contact
For any questions or suggestions, please open an issue in the repository or contact the maintainer directly at pidoko1@gmail.com.



