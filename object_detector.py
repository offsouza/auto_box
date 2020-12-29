#import logging
import os
import torch

import detect

# from log import get_logger

# logger = get_logger('detection_server')

class YOLO:
    def __init__(self, input_size=(416, 416), score=0.5, iou=0.5, device='cpu'):
        self.device = device
        self.score = score
        self.iou = iou       
        self.model_image_size = input_size
        self.is_fixed_size = self.model_image_size != (None, None)       
        self._load_model()
        #logger.info('YOLOv5s model successfully initialized')

    def _load_model(self):
        self.detector, self.names, self.colors, self.device, self.half = detect.load_model(device=self.device)
        #print(self.names)
    def _preprocess_image(self, image):        
        return detect.pre_process([image], self.model_image_size[0])

    def detect_object(self, image, cam, max_person):
        image_array = self._preprocess_image(image)
        detections = list()
        with torch.no_grad():
            detections = detect.detect(self.detector, self.names, self.colors, self.device, self.half, image_array, [image], [cam], max_person) 
        return detections


if __name__ == '__main__':
    import cv2

    detector = YOLO()
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     _, img = cap.read()
    #     print(detector.detect_object(img, 0))

    from glob import glob
    img_list = glob('/home/walysson/ProSecurity/Azure/Weapon-Detection-And-Classification/test/*')
    for im in img_list:
        img = cv2.imread(im)
        for i, detection in enumerate(detector.detect_object(img, 0)):
            bbox = detection[0]
            cv2.imwrite(im.replace('.', '_{}.'.format(i)), img[bbox[1]:bbox[3], bbox[0]:bbox[2], :])