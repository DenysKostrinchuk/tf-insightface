import preprocessor_worker
import argparse
import cv2
import sys
import numpy as np
import time
from models import base_server
from configs import configs

class FaceRecognition:

    def __init__(self,args):
        self.model = preprocessor_worker.FaceModel(args)
        
        self.feature_extractor = base_server.BaseServer(model_fp=configs.face_describer_model_fp,
                             input_tensor_names=configs.face_describer_input_tensor_names,
                             output_tensor_names=configs.face_describer_output_tensor_names,
                             device=configs.face_describer_device)
    def _normalize(self, x):
        n = np.linalg.norm(x, axis=-1)
        return x / np.expand_dims(n, -1)    
    
    
    def predict(self, img, tmp):
        
        start_detect = time.time()
        img = self.model.get_input(img)
        if np.any(img):
            input_data = np.array([np.expand_dims(img/255.0, axis=0)])
            features = self.feature_extractor.inference(data=input_data)
            features = self._normalize(np.array(features).reshape(-1, 512))

            #cv2.imwrite('test_cropped/{}.jpg'.format(self.idx),pure_img)
            #self.idx+=1
            #print('finish_features - {}'.format(time.time() - start_features))
        else:
            features = None
        return features,img