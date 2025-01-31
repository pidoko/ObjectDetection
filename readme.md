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
