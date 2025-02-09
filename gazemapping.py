"""
gazemapping.py
"""
import os
import sys
import json
import time
from datetime import datetime
import csv
import collections
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import uuid
from python_server import start_gaze_server, read_gaze_data  # Import gaze server
from PyQt6.QtCore import QTimer, Qt, QPoint, QObject, QEventLoop
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient
# from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import pyautogui  # For capturing screenshots
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run gaze mapping with optional settings.")
parser.add_argument(
    "--program_duration",
    type=int,
    default=60,  # Default to 10 seconds
    help="Set the duration of the program in seconds."
)
parser.add_argument(
    "--distraction_tolerance",
    type=int,
    default=5,  # Default to 10 seconds
    help="Set the duration in SECONDS for how long user can get distracted before providing feedback."
)


# Parse arguments
args = parser.parse_args()

# Use parsed values
program_duration = args.program_duration
distraction_tolerance = args.distraction_tolerance # seconds

print(f"Program Duration: {program_duration} seconds")
    
session_id = str(uuid.uuid4())[:8] # Generate unique session ID 

# Initialize PyQt Application
app = QApplication(sys.argv)
screen = app.primaryScreen()
screen_size = screen.size()
screen_width = screen_size.width()
screen_height = screen_size.height()
screen_midpoint = screen_width // 2

# Create an overlay window for the red circle
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                            Qt.WindowType.WindowStaysOnTopHint | 
                            Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(0, 0, screen_width, screen_height)
        self.circle_x = screen_width // 2
        self.circle_y = screen_height // 2
        self.show()
        self.show_calibration_points = False  # Flag to control calibration point visibility
        self.calibration_points = []  # List to store calibration point positions

    def update_position(self, x, y):
        self.circle_x = x
        self.circle_y = y
        self.repaint()
        self.update()
    
    def set_calibration_points(self):
        self.calibration_points = [
            (screen_width // 4, screen_height // 4),
            (3 * screen_width // 4, screen_height // 4),
            (screen_width // 4, 3 * screen_height // 4),
            (3 * screen_width // 4, 3 * screen_height // 4),
            (screen_width // 2, screen_height // 2)
        ]
        self.show_calibration_points = True  # Show the points
        self.update()

    def clear_calibration_points(self):
        self.calibration_points = []
        self.show_calibration_points = False  # Hide the points
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.show_calibration_points:
            pen = QPen(QColor(0, 255, 0))  # Green
            pen.setWidth(5)
            brush = QBrush(QColor(0, 255, 0)) # Green fill
            painter.setPen(pen)
            painter.setBrush(brush)
            for x, y in self.calibration_points:
                painter.drawEllipse(QPoint(int(x), int(y)), 10, 10)  # Draw green circles
        else:

            # painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # pen = QPen(QColor(255, 0, 0))
            # pen.setWidth(3)
            # painter.setPen(pen)
            # painter.setBrush(QColor(255, 0, 0, 127))
            # painter.drawEllipse(QPoint(self.circle_x, self.circle_y), 20, 20)

            # painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Create a dark overlay
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))  # Semi-transparent dark background

            # Create a spotlight effect using a radial gradient
            self.spotlight_radius = 250 
            gradient = QRadialGradient(self.circle_x, self.circle_y, self.spotlight_radius)
            gradient.setColorAt(0.0, QColor(255, 255, 255, 0))    # Fully transparent at center
            gradient.setColorAt(0.5, QColor(0, 0, 0, 25))         # Midpoint transition (lighter dark)
            gradient.setColorAt(0.7, QColor(0, 0, 0, 50))        # Steeper fade to dark
            gradient.setColorAt(1.0, QColor(0, 0, 0, 200))        # Fully dark at the edges


            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect())


# Start OpenFace server and overlay UI
overlay = Overlay()
client_socket, process = start_gaze_server()

# calibration_enabled = False
ONESHOT = True # if we take one screenshot at end, make true

# 1. Calibration Points:
CALIBRATION_POINTS = [
    (screen_width * 0.05, screen_height * 0.05),       # Very near top-left
    (screen_width * 0.95, screen_height * 0.05),       # Very near top-right
    (screen_width * 0.05, screen_height * 0.95),       # Very near bottom-left
    (screen_width * 0.95, screen_height * 0.95),       # Very near bottom-right
    (screen_width // 2, screen_height // 2),      # Center
    (screen_width // 2, screen_height * 0.05),       # Top-center
    (screen_width // 2,  screen_height * 0.95),   # Bottom-center
    (screen_width // 8, screen_height // 2),       # Left-center
    (7 * screen_width // 8, screen_height // 2)    # Right-center
]

# 2. Calibration Data Storage:
calibration_data = []  # List to store gaze data and x,y screen coordinates

def save_calibration_data(calibration_data, filename="calibration_data.csv"):
    file_exists = os.path.isfile(filename)
    
    # Define column order
    columns = [
        "session_id", "yaw", "pitch", 
        "gaze_left_x", "gaze_left_y", "gaze_left_z",
        "gaze_right_x", "gaze_right_y", "gaze_right_z",
        "head_tx", "head_ty", "head_tz",
        "head_roll", "head_pitch", "head_yaw",
        "screen_x", "screen_y"
    ]

    # Ensure calibration_data is always a list
    if isinstance(calibration_data, dict):
        calibration_data = [calibration_data]  # Convert single dictionary to a list

    for data in calibration_data:
        data["session_id"] = session_id  # Assign session ID to each row

    # Convert to DataFrame
    df = pd.DataFrame(calibration_data, columns=columns)

    # Append to CSV (create if it doesn't exist)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
    print(f"✅ Calibration data saved to {filename}")


# 3. Calibration Function:
def calibrate_gaze():
    # global calibration_data

    for point in CALIBRATION_POINTS:
        x, y = point

        # Show only the current calibration point
        overlay.calibration_points = [(x, y)]  # Display only one point at a time
        overlay.show_calibration_points = True  
        overlay.update()
        app.processEvents()  # Ensure UI updates immediately

        print(f"Look at the green dot for 4 seconds: ({x}, {y})")
        start_time = time.time()
        gaze_data_points = []

        while time.time() - start_time < 4:
            gaze_data = read_gaze_data(client_socket)
            if gaze_data and all(key in gaze_data for key in ['yaw', 'pitch', 'gaze_left', 'gaze_right', 'head_pose']):
                # Extract values
                yaw = gaze_data['yaw']
                pitch = gaze_data['pitch']
                gaze_left = gaze_data['gaze_left']
                gaze_right = gaze_data['gaze_right']
                head_pose = gaze_data['head_pose']

                # Check if any critical values are exactly 0.0 and abort
                if yaw == 0.0 or pitch == 0.0 or head_pose['yaw'] == 0.0 or head_pose['pitch'] == 0.0:
                    print(f"Aborting: Invalid gaze data detected (zero values): {gaze_data}")
                    return 
                
                # Collect more comprehensive data
                gaze_data_points.append({
                    'yaw': gaze_data['yaw'],
                    'pitch': gaze_data['pitch'],
                    'gaze_left_x': gaze_data['gaze_left']['x'],
                    'gaze_left_y': gaze_data['gaze_left']['y'],
                    'gaze_left_z': gaze_data['gaze_left']['z'],
                    'gaze_right_x': gaze_data['gaze_right']['x'],
                    'gaze_right_y': gaze_data['gaze_right']['y'],
                    'gaze_right_z': gaze_data['gaze_right']['z'],
                    'head_tx': gaze_data['head_pose']['tx'],
                    'head_ty': gaze_data['head_pose']['ty'],
                    'head_tz': gaze_data['head_pose']['tz'],
                    'head_roll': gaze_data['head_pose']['roll'],
                    'head_pitch': gaze_data['head_pose']['pitch'],
                    'head_yaw': gaze_data['head_pose']['yaw']
                })
            app.processEvents() # Important: Process events to keep the GUI responsive

        if gaze_data_points:
            avg_data = {
                key: np.median([point[key] for point in gaze_data_points])
                for key in gaze_data_points[0].keys()
            }
            avg_data['screen_x'] = int(x)
            avg_data['screen_y'] = int(y)
            
            calibration_data.append(avg_data)
            save_calibration_data(avg_data)
            print(f"Calibration data added for point: ({x}, {y})")
        else:
            print("No gaze data received for this point.")

        # Clear the calibration point before moving to the next point
        overlay.calibration_points = []
        overlay.update()
        app.processEvents()

    overlay.clear_calibration_points()  # Hide calibration points
    return calibration_data


# Wait for OpenFace to start streaming data before calibrating
print("Waiting for OpenFace gaze data...")

while True:
    gaze_data = read_gaze_data(client_socket)
    if gaze_data and gaze_data.get("yaw") is not None and gaze_data.get("pitch") is not None:
        print("✅ OpenFace data streaming detected. Starting calibration...")
        break  # Exit loop when we receive valid gaze data

    time.sleep(0.1)  # Small delay to prevent CPU overload


# 4. Call Calibration Function:
calibration_enabled = False
if calibration_enabled:
    calibration_data = calibrate_gaze() # Run before main loop
    print("Calibration Complete:", calibration_data)


def train_model_from_csv(filename="calibration_data.csv"):
    # Load data
    if not os.path.isfile(filename):
        print("No calibration data available.")
        return None, None

    df = pd.read_csv(filename)

    # Define feature columns
    features = ["yaw", "pitch", 
                "gaze_left_x", "gaze_left_y", "gaze_left_z",
                "gaze_right_x", "gaze_right_y", "gaze_right_z",
                "head_tx", "head_ty", "head_tz",
                "head_roll", "head_pitch", "head_yaw"]

    X = df[features]
    y_x = df["screen_x"]
    y_y = df["screen_y"]

    # # Train models
    # x_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    # y_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    x_model = LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
    y_model = LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)


    x_model.fit(X, y_x)
    y_model.fit(X, y_y)

    print("✅ Models trained on calibration data.")
    return x_model, y_model

x_model, y_model = train_model_from_csv()

def map_gaze_to_screen(gaze_data):
    if not x_model or not y_model:
        print("No trained model available, using default mapping.")
        SCALING_FACTOR = 400
        screen_x = int(screen_width / 2 + gaze_data["yaw"] * SCALING_FACTOR)
        screen_y = int(screen_height / 2 - gaze_data["pitch"] * SCALING_FACTOR)
        return screen_x, screen_y

    # Prepare input features
    features = ["yaw", "pitch", 
                "gaze_left_x", "gaze_left_y", "gaze_left_z",
                "gaze_right_x", "gaze_right_y", "gaze_right_z",
                "head_tx", "head_ty", "head_tz",
                "head_roll", "head_pitch", "head_yaw"]

    # Convert gaze data into a DataFrame so feature names are preserved
    current_features_df = pd.DataFrame([gaze_data], columns=features)

    # Predict screen coordinates
    screen_x = int(x_model.predict(current_features_df)[0])
    screen_y = int(y_model.predict(current_features_df)[0])

    # Clamping to screen bounds
    screen_x = max(0, min(screen_x, screen_width - 1))
    screen_y = max(0, min(screen_y, screen_height - 1))

    return screen_x, screen_y



# Define filename for gaze tracking data
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
GAZE_LOG_FILE = f"gaze_tracking_{timestamp_str}.csv"

# Define how often to take screenshots
SCREENSHOT_INTERVAL = 10 # seconds
screenshot_data = []  # Store screenshot metadata

# Directory to save screenshots
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Define filename for screenshot metadata
SCREENSHOT_METADATA_FILE = f"screenshot_metadata_{timestamp_str}.csv"

# Screenshot metadata buffer
screenshot_buffer = []
SCREENSHOT_BUFFER_SIZE = 5 



# Function to save screenshot metadata in batches
def save_screenshot_data():
    global screenshot_buffer
    if not screenshot_buffer:
        return  # Don't write if buffer is empty

    file_exists = os.path.isfile(SCREENSHOT_METADATA_FILE)

    with open(SCREENSHOT_METADATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "screenshot_path"])

        # Write buffered screenshot metadata
        writer.writerows(screenshot_buffer)
    
    print(f"✅ {len(screenshot_buffer)} screenshot metadata records saved to {SCREENSHOT_METADATA_FILE}")
    screenshot_buffer.clear()  # Clear buffer after writing

screenshot_filename = ""

# Function to take a screenshot and store metadata in buffer
def take_screenshot():
    global last_screenshot_time
    global screenshot_filename

    timestamp = time.time()  # Get precise timestamp
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Human-readable
    screenshot_filename = f"screenshot_{dt_string}.png"
    filepath = os.path.join(screenshot_dir, screenshot_filename)

    # Capture screenshot
    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)

    # Store metadata in buffer
    screenshot_buffer.append([timestamp, filepath])
    print(f"📸 Screenshot saved: {filepath}")

    # Save to CSV when buffer reaches its limit
    if len(screenshot_buffer) >= SCREENSHOT_BUFFER_SIZE or ONESHOT:
        save_screenshot_data()


# Function to save gaze data to CSV
# def save_gaze_data(timestamp, screen_x, screen_y, session_id):
def save_gaze_data():
    global gaze_buffer
    if not gaze_buffer:
        return # don't write if buffer empty

    file_exists = os.path.isfile(GAZE_LOG_FILE)
    
    # Open CSV file in append mode
    with open(GAZE_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["session_id", "timestamp", "screen_x", "screen_y"])

        # Write gaze data rows
        writer.writerows(gaze_buffer)
        # writer.writerow([session_id, timestamp, screen_x, screen_y])
    print(f"✅ {len(gaze_buffer)} gaze records saved to {GAZE_LOG_FILE}")
    gaze_buffer.clear()  # Clear buffer after writing

# For moving averaging (SMOOTHING)
N = 25  # Number of frames to average over (HIGHER IS SMOOTHER)
gaze_positions = collections.deque(maxlen=N)
gaze_buffer = []
BUFFER_SIZE = 30  # Write every 30 frames (about once per second)

last_screenshot_time = time.time() # init screenshot time

distracted_counter = 0

# MAIN LOOP AFTER CALIBRATION
def update_gaze():
    global last_screenshot_time
    global distracted_counter

    gaze_data = read_gaze_data(client_socket)
    # print("Gaze Data:", gaze_data)

    if gaze_data and all(key in gaze_data for key in ['yaw', 'pitch', 'gaze_left', 'gaze_right', 'head_pose']):
        # Prepare full feature dictionary
        full_gaze_data = {
            'yaw': gaze_data['yaw'],
            'pitch': gaze_data['pitch'],
            'gaze_left_x': gaze_data['gaze_left']['x'],
            'gaze_left_y': gaze_data['gaze_left']['y'],
            'gaze_left_z': gaze_data['gaze_left']['z'],
            'gaze_right_x': gaze_data['gaze_right']['x'],
            'gaze_right_y': gaze_data['gaze_right']['y'],
            'gaze_right_z': gaze_data['gaze_right']['z'],
            'head_tx': gaze_data['head_pose']['tx'],
            'head_ty': gaze_data['head_pose']['ty'],
            'head_tz': gaze_data['head_pose']['tz'],
            'head_roll': gaze_data['head_pose']['roll'],
            'head_pitch': gaze_data['head_pose']['pitch'],
            'head_yaw': gaze_data['head_pose']['yaw']
        }

        x, y = map_gaze_to_screen(full_gaze_data)

        # 
        if x < screen_midpoint:
            distracted_counter += 1
        else:
            distracted_counter = 0

        if distracted_counter >= distraction_tolerance * 33:
            print("DISTRACTED")
            # WRITE TO JSON FILE
            # Prepare JSON output (all values converted to Python-native types)
            distracted_JSON = {
                "distracted": True,  
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Human-readable timestamp
                "epoch_time": time.time(),
            }

            # Write to a JSON file
            json_filename = f"distracted.json"
            with open(json_filename, "w") as json_file:
                json.dump(distracted_JSON, json_file, indent=4)

            print(f"📄 Session analysis saved to {json_filename}")



            distracted_counter = -100


        # SAVE TO CSV
        # Get timestamp in seconds
        timestamp = time.time()

        gaze_buffer.append([session_id, timestamp, x, y])

        # Write to CSV every BUFFER_SIZE frames
        if len(gaze_buffer) >= BUFFER_SIZE:
            save_gaze_data()  # Write batch to CSV
        # save_gaze_data(timestamp, x, y, session_id)

        # Capture screenshots every SCREENSHOT_INTERVAL seconds
        if not ONESHOT and timestamp - last_screenshot_time >= SCREENSHOT_INTERVAL:
            take_screenshot()
            last_screenshot_time = timestamp

        # print(f"Screen X: {x}, Screen Y: {y}, Timestamp: {timestamp}")

        gaze_positions.append((x, y))

        # Calculate moving average
        avg_x = sum(pos[0] for pos in gaze_positions) / len(gaze_positions)
        avg_y = sum(pos[1] for pos in gaze_positions) / len(gaze_positions)

        overlay.update_position(int(avg_x), int(avg_y))
    else:
        print("Incomplete gaze data")

# Set up a QTimer to update the overlay 30 times per second
timer = QTimer()
timer.timeout.connect(update_gaze)
timer.start(33)

overlay.show()

QTimer.singleShot(program_duration * 1000, app.quit) # HOW LONG THE PROGRAM RUNS

# Run the PyQt event loop
exit_code = app.exec()

take_screenshot()

# Post-session analysis
# read session data csv
df_output = pd.read_csv(GAZE_LOG_FILE)
# Calculate percentage of points on the left side of the screen vs right
left_side_count = int((df_output['screen_x'] < screen_midpoint).sum())  # Convert to Python int
right_side_count = int((df_output['screen_x'] >= screen_midpoint).sum())  # Convert to Python int

total_time = int(len(df_output))  # Ensure it's a Python int
left_ratio = float(left_side_count / total_time) if total_time > 0 else 0.0
right_ratio = float(right_side_count / total_time) if total_time > 0 else 0.0

# Prepare JSON output (all values converted to Python-native types)
session_analysis = {
    "session_id": str(session_id),  # Ensure string type
    "total_time": int(program_duration),
    "left_ratio": left_ratio,
    "right_ratio": right_ratio,
    "left_count": left_side_count,
    "right_count": right_side_count
}

# Write to a JSON file
json_filename = f"session_analysis_{session_id}.json"
with open(json_filename, "w") as json_file:
    json.dump(session_analysis, json_file, indent=4)

print(f"📄 Session analysis saved to {json_filename}")

# Generate heatmap overlay image

# Construct the screenshot path
screenshot_path = f"screenshots/{screenshot_filename}"

# Verify if the file exists
if not os.path.exists(screenshot_path):
    print(f"❌ ERROR: Screenshot file not found: {screenshot_path}")
else:
    background = cv2.imread(screenshot_path)
    if background is None:
        print(f"❌ ERROR: cv2.imread() failed to read {screenshot_path}")
    else:
        screenshot_height, screenshot_width, _ = background.shape
expected_width = 1710
expected_height = 1107
screenshot_height, screenshot_width, _ = background.shape
if screenshot_width == expected_width * 2 and screenshot_height == expected_height * 2:
    background = cv2.resize(background, (expected_width, expected_height), 
                          interpolation=cv2.INTER_AREA)
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
background = np.flip(background, axis=0)  # Vertical flip
heatmap, x_edges, y_edges = np.histogram2d(
    df_output['screen_x'], 
    df_output['screen_y'], 
    bins=[screen_width // 10, screen_height // 10]
)
heatmap = gaussian_filter(heatmap, sigma=6)
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
# heatmap = np.flip(heatmap, axis=1)  # Flip heatmap horizontally
# heatmap = np.flip(heatmap, axis=0)  # Flip heatmap horizontally
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(
    background,
    extent=[0, heatmap.shape[0], 0, heatmap.shape[1]],
    alpha=0.6
)
sns.heatmap(
    heatmap.T,
    cmap='inferno',
    alpha=0.5,
    xticklabels=False,
    yticklabels=False,
    cbar=False,
    ax=ax
)
plt.axis("off")
plt.savefig(
    "screenshots/gaze_heatmap_overlay.png", 
    dpi=300, 
    bbox_inches='tight', 
    pad_inches=0
)


# Cleanup when exiting
save_screenshot_data()  # Ensure any remaining screenshot metadata is written
save_gaze_data()

# Cleanup when exiting
client_socket.close()
process.terminate()
sys.exit(exit_code)