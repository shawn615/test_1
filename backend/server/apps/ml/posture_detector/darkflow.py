# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:21:08 2020

@author: Seunghyeon Seo
"""

from darkflow.net.build import TFNet
import cv2
import os
import json

class DarkflowDetector:
    def __init__(self, pbpath, metapath):
        self.options = {"pbLoad": pbpath, "metaLoad": metapath, "threshold": 0.25} # model path
        self.tfnet = TFNet(self.options)
        
    def detect(self, input_data):
        self.frame = cv2.imread(input_data)
        self.root, self.file_name = os.path.split(input_data)
        
        self.results = self.tfnet.return_predict(self.frame)
        
        for result in self.results:
            result['confidence'] = str(result['confidence'])
        #     name = result["label"]
        #     xmin = result["topleft"]["x"]
        #     ymin = result["topleft"]["y"]
        #     xmax = result["bottomright"]["x"]
        #     ymax = result["bottomright"]["y"]
        #     x_mid = (xmin + xmax) / 2
        #     y_mid = (ymin + ymax) / 2
        #     confidence = result["confidence"]
        #     text_position = ymin - 10
        #     if text_position < 0:
        #         text_position = 10
    
        #     r = 0
        #     g = 0
        #     b = 0
    
        #     if "Korean_Cow" in name or "standing" in name:
        #         r = 0
        #         g = 176
        #         b = 80
        #     elif "Korean_Calf" in name or "lying" in name:
        #         r = 255
        #         g = 255
        #         b = 0
        #     else:
        #         r = 192
        #         g = 0
        #         b = 0
                
        #     self.frame = cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (b,g,r), 2)
        #     cv2.putText(self.frame, name, (xmin, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (b,g,r), 2)
        #     cv2.putText(self.frame, str(round(confidence,2)), (xmin, text_position-23), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (b,g,r), 2)
        
        # cv2.imwrite('{}_detected.jpg'.format(self.file_name.replace('.jpg','')), frame)
        # or
        
        return json.dumps(self.results)
    
    # def save(self, path):
    #     cv2.imwrite("{}/{}_detected.jpg".format(path, self.file_name.replace('.jpg','')), self.frame)

