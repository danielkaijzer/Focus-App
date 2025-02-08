"""
python_server.py
"""

import socket
import subprocess
import time
import json

# Set the absolute path to FaceLandmarkVid
FACELANDMARKVID_PATH = "/Users/danielkaijzer/Desktop/Cornell_AI_Hackathon/Focus_App/external_libs/openFace/OpenFace/build/bin/FaceLandmarkVid"


def start_gaze_server():
    """Starts the OpenFace gaze tracking process and returns a socket connection."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 5005))
    server_socket.listen(1)
    print("Python server is listening on port 5005...")

    # Ensure server is ready before starting OpenFace
    time.sleep(2)
    process = subprocess.Popen([FACELANDMARKVID_PATH, "-device", "0", "-gaze", "-pose", "-verbose"])

    # Accept connection from C++ OpenFace process
    client_socket, addr = server_socket.accept()
    print(f"Connected to {addr}")

    return client_socket, process

def read_gaze_data(client_socket):
    """Reads and returns gaze data from OpenFace."""
    try:
        data = client_socket.recv(1024).decode()
        if not data:
            return None
        return json.loads(data)  # Convert JSON string to Python dictionary
    except json.JSONDecodeError:
        print("Error decoding JSON")
        return None