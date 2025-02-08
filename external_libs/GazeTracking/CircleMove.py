import sys
import os
import platform
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen
from gaze_tracking import GazeTracking

# Fix for macOS Cocoa plugin
if platform.system() == "Darwin":
    import site
    base_path = site.getsitepackages()[0]
    plugin_path = os.path.join(base_path, "PyQt6/Qt6/plugins/platforms")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

class GazeOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)
        self.smooth_buffer = np.zeros((10, 2))  # Smoothing over 10 frames
        self.raw_gaze = (0.5, 0.5)  # Default normalized center position
        
        # Window setup: full-screen overlay
        screen = QApplication.primaryScreen().size()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.position = QPoint(screen.width() // 2, screen.height() // 2)

        # Timer for updating frames (~33 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Draw the red circle (gaze cursor)
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.setBrush(QColor(255, 0, 0, 127))
        painter.drawEllipse(self.position, 15, 15)

    def process_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            return

        try:
            self.gaze.refresh(frame)
            left_pupil = self.gaze.pupil_left_coords()
            right_pupil = self.gaze.pupil_right_coords()
            if left_pupil and right_pupil:
                # Average the pupil positions
                avg_x = (left_pupil[0] + right_pupil[0]) / 2
                avg_y = (left_pupil[1] + right_pupil[1]) / 2
                # Normalize based on frame size (values between 0 and 1)
                self.raw_gaze = (avg_x / frame.shape[1], avg_y / frame.shape[0])
                # Debug: print raw gaze values
                print(f"Raw Gaze: {self.raw_gaze}")
        except Exception as e:
            print(f"Error processing frame: {e}")

        self.update_position()
        self.update()  # trigger repaint

    def update_position(self):
        # Direct mapping from normalized gaze to screen coordinates
        flipped_x = 1.0 - self.raw_gaze[0]
        new_point = np.array([flipped_x, self.raw_gaze[1]])
        
        # Update smoothing buffer and compute median
        self.smooth_buffer = np.roll(self.smooth_buffer, -1, axis=0)
        self.smooth_buffer[-1] = new_point
        smoothed = np.median(self.smooth_buffer, axis=0)

        # Map normalized values to pixel coordinates
        x_pixel = int(smoothed[0] * self.width())
        y_pixel = int(smoothed[1] * self.height())

        # Ensure the circle stays within screen boundaries
        x_pixel = max(0, min(self.width(), x_pixel))
        y_pixel = max(0, min(self.height(), y_pixel))
        self.position.setX(x_pixel)
        self.position.setY(y_pixel)
        # Debug: print screen coordinates
        print(f"Screen Position: {x_pixel}, {y_pixel}")

    def closeEvent(self, event):
        self.webcam.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = GazeOverlay()
    overlay.show()
    sys.exit(app.exec())