"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.posture_detector.darkflow import DarkflowDetector

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    df = DarkflowDetector(pbpath="D:\code\Yangpyong_Detecting_Model/tiny-yolo-voc-custom.pb", metapath="D:\code\Yangpyong_Detecting_Model/tiny-yolo-voc-custom.meta")
    # add to ML registry
    registry.add_algorithm(endpoint_name="posture_detector",
                            algorithm_object=df,
                            algorithm_name="darkflow",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Shawn",
                            algorithm_description="Object Detecting and Image Saving by Darkflow",
                            algorithm_code=inspect.getsource(DarkflowDetector))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))