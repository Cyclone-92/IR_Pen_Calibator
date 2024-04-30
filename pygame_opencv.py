import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
import pygame
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtWidgets
import sys
import pickle
from sklearn.preprocessing import OneHotEncoder
import gc
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from keras.utils import to_categorical
from keras.models import load_model
import pickle

class PyGame_thread(QThread):

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.image = 0
        self.stop_pygame = False
        self.pygame_end = False
        self.select_mode = ""
        with open('encoder.pickle', 'rb') as f:
            # Use pickle.load() to load the data
            self.encode = pickle.load(f)
        self.model = load_model("12_classes.h5")
        

    def select_mode_func(self,mode):
        self.select_mode = mode

    def stop_pygame_functions(self):
        self.stop_pygame = True

    def run(self):
        print(f"selected thread mode {self.select_mode}")
        if self.select_mode == "IR_Pen":
            self.pygame_code_ir()
        elif self.select_mode == "Mouse":
            self.pygame_mouse()
        print("Ending the pygame")
        self.quit()

    def pygame_code_ir(self):
        pygame.init()
        main_count = 0
        # Set the width and height of the self.screen (adjust as needed)
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.font = pygame.font.Font(None, 36)
        # Set up the self.screen
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Drawing with Pygame")
        
        # Set colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red   = (255, 0, 0)
        self.green = (0,255,0)
        # Set circle parameters
        circle_radius = 20  # Adjust as needed

        # set medssage
        pygame.display.set_caption("Text Display Example")

        while(not self.stop_pygame):

            # Convert frame to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Thresholding to isolate bright areas (IR light)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate through contours
            for contour in contours:
                # Compute centroid of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Draw centroid on the frame
                    # cv2.circle(self.image, (cX, cY), 5, (0, 0, 255), -1)
                    # Print normalized coordinates
                    video_height, video_width, _ = self.image.shape
                    normalized_cX = cX / video_width
                    normalized_cY = cY / video_height

                    # # Display frame
                    # cv2.imshow('IR Detection', frame)
                    
                    # Draw circle at the centroid in Pygame
                    # cv2.imshow('IR Detection',self.image)
                    pygame.draw.circle(self.screen, self.white, (int(cX * self.SCREEN_WIDTH / self.image.shape[1]), int(cY * self.SCREEN_HEIGHT / self.image.shape[0])), circle_radius)
            
            
            if main_count == 0:
                self.main_message()
            # Update the display
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: 
                        self.pygame_end = True
                        pygame.quit()
                    elif event.key == pygame.K_c:
                        self.screen.fill(self.black)
                    elif event.key == pygame.K_RETURN:
                        main_count += 1
                        if main_count != 0:
                            self.screen.fill(self.black)
                    elif event.key == pygame.K_s:
                        pygame.image.save(self.screen, "snapshot.jpg")
                        animal,confidence = self.preprocessing()
                        self.screen.fill(self.black)
                        self.predict(animal,confidence)
        print("Pygame exit check")
        pygame.quit()

    def pygame_mouse(self):
        # Initialize Pygame
        pygame.init()

        # Set the width and height of the self.screen (adjust as needed)
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Drawing with Mouse")
        self.font = pygame.font.Font(None, 36)
        # Define colors
        # Set colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red   = (255, 0, 0)
        self.green = (0,255,0)

        # Set up drawing variables
        drawing = False
        last_pos = None
        clock = pygame.time.Clock()

        # Set circle parameters
        circle_radius = 20  # Adjust as needed

        # Main loop
        while not self.stop_pygame:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: 
                        self.pygame_end = True
                        pygame.quit()
                    elif event.key == pygame.K_c:
                        self.screen.fill(self.black)
                    elif event.key == pygame.K_RETURN:
                        main_count += 1
                        if main_count != 0:
                            self.screen.fill(self.black)
                    elif event.key == pygame.K_s:
                        pygame.image.save(self.screen, "snapshot.jpg")
                        animal,confidence = self.preprocessing()
                        self.screen.fill(self.black)
                        self.predict(animal,confidence)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        drawing = True
                        last_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        drawing = False
                        last_pos = None
                elif event.type == pygame.MOUSEMOTION:
                    if drawing:
                        current_pos = event.pos
                        if last_pos:
                            pygame.draw.line(self.screen, self.white, last_pos, current_pos, circle_radius)
                        last_pos = current_pos

            # Update the display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(60)

    def main_message(self):
        # Set font
        
        # Render text onto a surface
        text_surface = self.font.render('Hello, Lets get drawing!\n press Enter to begin.', True, self.red)  # Change the text as needed

        # Blit the text surface onto the self.screen
        self.screen.blit(text_surface, (self.SCREEN_WIDTH // 2 - text_surface.get_width() // 2, self.SCREEN_HEIGHT // 2 - text_surface.get_height() // 2))

    def predict(self,animal,confidence):
        # Render text onto a surface
        text_surface = self.font.render(f'Did you draw a {animal} ? ({confidence}%)', True, self.green)  # Change the text as needed
        # Blit the text surface onto the self.screen
        self.screen.blit(text_surface, (self.SCREEN_WIDTH // 2 - text_surface.get_width() // 2, self.SCREEN_HEIGHT // 2 - text_surface.get_height() // 2))

    def preprocessing(self):
        image = cv2.imread("snapshot.jpg")
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        cropped = self.image_cropper(image)
        kernel1 = np.ones((20, 20), np.uint8)
        closed_image = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel1)
        image = cv2.resize(closed_image, (28, 28))
        image = np.reshape(image, (28,28,1))
        cv2.imwrite("preprocessing.jpg",image)
        input_image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model.predict(input_image)
        max_index = np.argmax(prediction)
        one_hot_encoded = np.zeros_like(prediction)
        one_hot_encoded[0][max_index] = 1

        confidence = "{:.2f}".format((prediction[0][max_index])*100)
        print(confidence)
        animal = self.encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]
        print(f"prediction is {animal} confidence is {confidence}")
        return animal,confidence

    def image_cropper(self, image):
        image = cv2.imread("snapshot.jpg", 0)
        image[image>0] = 255
        non_zero = np.nonzero(image)
        box = [(min(x),max(x)+1) for x in non_zero]
        slices = [slice(*a) for a in box]
        result = image[tuple(slices)]
        return result

if __name__ == "__main__":
    initialize = PyGame_thread()
    initialize.start()
    # sys.exit(initialize.py_app.exec())