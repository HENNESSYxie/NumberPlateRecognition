import cv2

from detector.YOLOv5PlateDetector import YOLOv5PlateDetector
from aligner.EfficientNetV2sAligner import EfficientNetV2sAligner
from visualizer.Visualizer import Visualizer
from decoder.VideoDecoder import VideoDecoder

visualizer = Visualizer()
detector = YOLOv5PlateDetector(model="/home/hennessy/Desktop/NumberPlateRecognition/NumberPlateRecognition/assets/models/detector/YOLOv5OpenVINO/yolov5.xml",
                          weights="/home/hennessy/Desktop/NumberPlateRecognition/NumberPlateRecognition/assets/models/detector/YOLOv5OpenVINO/yolov5.bin",
                          output_layer_name="output0",
                          score_threshold=0.7,
                          nms_threshold=0.7,
                          width=640,
                          height=640)
aligner = EfficientNetV2sAligner(model="/home/hennessy/Desktop/NumberPlateRecognition/NumberPlateRecognition/assets/models/aligner/EfficientNetV2s/efficientnetv2s_openvino.xml",
                                 weights="/home/hennessy/Desktop/NumberPlateRecognition/NumberPlateRecognition/assets/models/aligner/EfficientNetV2s/efficientnetv2s_openvino.bin",
                                 output_layer_name="Func/StatefulPartitionedCall/output/_679:0",
                                 score_threshold=1,
                                 nms_threshold=1,
                                 width=224,
                                 height=224)

decoder = VideoDecoder("/windows/car_number_plates/videos/second.mp4")

for frame in decoder.decode():
    detections = detector.predict(frame)
    img = visualizer.visualize(image=frame, detections=detections)
    for det in detections:
        cropped = frame[det.y1:det.y2, det.x1:det.x2]
        warped = aligner.predict(cropped)
    # cv2.imshow('asd', img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
