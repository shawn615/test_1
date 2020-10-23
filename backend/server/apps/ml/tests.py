# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:41:35 2020

@author: Seunghyeon Seo
"""

from django.test import TestCase

from apps.ml.posture_detector.darkflow import DarkflowDetector

import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_dd_algorithm(self):
        input_data = "D:\Yangpyong\(Think) 20200601_PigImages/8_0_1_20200526060000_0.jpg"
        my_alg = DarkflowDetector(pbpath="D:\code\Yangpyong_Detecting_Model/tiny-yolo-voc-custom.pb", metapath="D:\code\Yangpyong_Detecting_Model/tiny-yolo-voc-custom.meta")
        response = my_alg.detect(input_data)
        # self.assertEqual('OK', response['status'])
        # self.assertTrue('label' in response)
        # self.assertEqual('<=50K', response['label'])
        
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "posture_detector"
        algorithm_object = DarkflowDetector(pbpath="D:\code\Yangpyong_Detecting_Model/tiny-yolo-voc-custom.pb", metapath="D:\code\Yangpyong_Detecting_Model/tiny-yolo-voc-custom.meta")
        algorithm_name = "darkflow"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Shawn"
        algorithm_description = "Object Detecting and Image Saving by Darkflow"
        algorithm_code = inspect.getsource(DarkflowDetector)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)