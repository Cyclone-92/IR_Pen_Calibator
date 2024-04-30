import sys
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QMainWindow,QGraphicsScene
from PySide6.QtGui import QPixmap,QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Signal
from cameras import get_available_cameras
import numpy as np
from camera_opncv import image_thread



class capstone():

    def __init__(self):
        self.loader = QUiLoader()
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = self.loader.load("untitled.ui")
        self.radiobutton = self.window.radiobutton
        self.comboBox = self.window.comboBox
        self.stop_button = self.window.stop_button
        self.slider = self.window.slider_
        self.startbutton = self.window.start_button
        self.textbox = self.window.textbox
        self.opencv_thread = image_thread()
    # ---------- Graphic Box --------------------
        self.graphicview = self.window.graphicview
        self.graphic_width = self.graphicview.size().width()
        self.graphic_height = self.graphicview.size().width()
        self.graphic_color_type = 3
        self.stop_signal = False
    # ---------- connection --------------------------
        self.radiobutton.setChecked(False)
        self.radiobutton.toggled.connect(self.on_radio_button_toggled)
        self.startbutton.clicked.connect(self.run_start)
        self.stop_button.clicked.connect(self.stop_all)
        self.opencv_thread = image_thread()
    
    def stop_all(self):
        self.opencv_thread.stop()
        if (self.comboBox.currentText() != "No camera selected"):
            self.graphicview.scene().clear()
        # self.opencv_thread.stop_signal.emit(True)
        

    def on_radio_button_toggled(self):
        if self.radiobutton.isChecked():
            self.update_output_terminal("Searching for cameras")
            cameras = get_available_cameras()
            self.comboBox.addItems(cameras)
        else:
            self.comboBox.clear()
            self.comboBox.addItem("No camera selected")
            self.update_output_terminal("stop searching cameras")
    
    def run_thread(self):
        current_camera_index = self.window.comboBox.currentIndex()
        camera_name = self.window.comboBox.itemText(current_camera_index)
        self.opencv_thread.camera_index = camera_name
        self.opencv_thread.height = self.graphic_height
        self.opencv_thread.width = self.graphic_width
        self.update_output_terminal("Video Broadcasting")
        self.opencv_thread.message_updated.connect(self.update_output_terminal)
        self.opencv_thread.image_updated.connect(self.render_graphics)
        self.opencv_thread.start()

    def update_output_terminal(self,mssg):
        self.textbox.setText(mssg)

    def run_start(self):
        if(self.comboBox.currentText() == "No camera selected"):
            self.update_output_terminal("Please select the camera")
        else:
            self.opencv_thread.not_stoped = False
            self.run_thread()

    def render_graphics(self,image):
        # Convert the numpy array image to a QImage
        q_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888).rgbSwapped()

        # Create a QGraphicsPixmapItem with the QImage
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))

        # Get the scene associated with the QGraphicsView
        scene = self.graphicview.scene()

        # If no scene exists, create a new one
        if scene is None:
            scene = QGraphicsScene()
            self.graphicview.setScene(scene)

        # Clear the scene
        scene.clear()

        # Add the QGraphicsPixmapItem to the scene
        scene.addItem(pixmap_item)

if __name__ == "__main__":
    start_capston = capstone()
    start_capston.window.show()
    start_capston.app.exec()