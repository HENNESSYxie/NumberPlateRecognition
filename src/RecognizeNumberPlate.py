import cv2

from detector.YOLOv5PlateDetector import YOLOv5PlateDetector
from visualizer.Visualizer import Visualizer

visualizer = Visualizer()
det = YOLOv5PlateDetector(model="/home/hennessy/Desktop/NumberPlateRecognition/NumberPlateRecognition/assets/models/detector/yolov5.xml",
                          weights="/home/hennessy/Desktop/NumberPlateRecognition/NumberPlateRecognition/assets/models/detector/yolov5.bin",
                          output_layer_name="output0",
                          score_threshold=0.7,
                          nms_threshold=0.7,
                          width=640,
                          height=640)
image = cv2.imread("/windows/car_number_plates/all_images/A084KK198.jpg")
img = visualizer.visualize(image=image, detections=det.predict(image))
cv2.imshow('asd', img)
cv2.waitKey(0)