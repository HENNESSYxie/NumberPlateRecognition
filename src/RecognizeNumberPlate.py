import cv2
import os

from detector.YOLOv5PlateDetector import YOLOv5PlateDetector
from aligner.EfficientNetV2sAligner import EfficientNetV2sAligner
from recognizer.PaddleOCR import PaddleOCR
from visualizer.Visualizer import Visualizer
from decoder.VideoDecoder import VideoDecoder
from tracker.SORTTracker import SORTTracker

os.environ['BASE_PATH'] = "/home/hennessy/Desktop/train_files/NumberPlateRecognition"
base_path = os.environ['BASE_PATH']
visualizer = Visualizer()
detector = YOLOv5PlateDetector(model=base_path + "/assets/models/detector/YOLOv5OpenVINO/yolov5.xml",
                               weights=base_path + "/assets/models/detector/YOLOv5OpenVINO/yolov5.bin",
                               output_layer_name="output0",
                               score_threshold=0.5,
                               nms_threshold=0.7,
                               width=640,
                               height=640)
aligner = EfficientNetV2sAligner(model="/home/hennessy/Desktop/MODELS/EfficientNetV2s/efficientnetv2s_openvino.xml",
                                 weights="/home/hennessy/Desktop/MODELS/EfficientNetV2s/efficientnetv2s_openvino.bin",
                                 output_layer_name="Func/StatefulPartitionedCall/output/_679:0",
                                 score_threshold=1,
                                 nms_threshold=1,
                                 width=224,
                                 height=224)
recognizer = PaddleOCR(model=base_path + "/assets/models/recognizer/PaddleOCR/paddleOCR.pdmodel",
                       weights=base_path + "/assets/models/recognizer/PaddleOCR/paddleOCR.pdiparams",
                       output_layer_name='save_infer_model/scale_0.tmp_1',
                       score_threshold=0.8,
                       nms_threshold=0.77,
                       width=320,
                       height=48)
video = "/windows/car_number_plates/videos/second.mp4"
decoder = VideoDecoder(video)
tracker = SORTTracker(min_hits=3, max_age=5)

for frame in decoder.decode():
    detections = detector.predict(frame)
    tracks = tracker.update(detections)
    for det in tracks:
        cropped = frame[det.detection.y1:det.detection.y2, det.detection.x1:det.detection.x2]
        warped = aligner.predict(cropped)
        warped = cv2.blur(warped, (1, 2))
        number = recognizer.predict(warped)
        frame = visualizer.visualize(image=frame, track=det, number=number)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
