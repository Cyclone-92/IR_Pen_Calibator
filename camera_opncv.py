import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class image_thread(QThread):

    image_updated = Signal(np.ndarray)  # Signal to indicate image update
    message_updated = Signal(str)
    
    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.width = 0
        self.height = 0
        self.channel = 3
        self.not_stoped = False

    def stop(self):
            self.not_stoped = True

    def run(self):
        while(not self.not_stoped):
            self.main_camcv()
    def emit_message(self,message):
        self.message_updated.emit(message)  # Emit the output message
    def emit_image(self, image):
        self.image_updated.emit(image)  # Emit the output message

    def main_camcv(self):


        cap = cv2.VideoCapture(f"/dev/{self.camera_index}")  # Open the selected camera
        
        if not cap.isOpened():
            self.emit_message("Error: Failed to open camera.")
            return

        while not self.not_stoped:
            # print(f"width {width} height {height} channel {channel} stop is {stop}")
            # print(f"not stop {not_stoped}")
            ret, frame = cap.read()  # Read a frame from the camera
            if (not ret):
                self.emit_message("Error: Failed to read frame.")
                break
            elif(self.not_stoped):
                self.emit_message("Broadcasting ended")
                break
            
            # Lets resize to fit in the image
            resized_image =cv2.resize(frame,(self.width,self.height))
            # Lets flip the image
            flipped = cv2.flip(resized_image, 0)
            # Emit the frame as a signal
            self.emit_image(flipped)
            # print(frame.dtype)

            # You can add any processing or analysis of the frame here
            
        cap.release()

if __name__ == "__main__":
    initialize = image_thread()