import numpy as np
import json
import cv2
from  collections import OrderedDict
from scipy.spatial import distance
from FaceRecognition import FaceRecognition
import time



class Worker:
    def __init__(self, args):
        self.id_to_feat = OrderedDict()
        self.tr_pos=0
        self.tr_neg=0
        self.f_pos=0
        self.f_neg=0
        self.threshold = args.threshold
        self.wrong_list = []
        self.step_save_accuracy = args.step_accuracy
        self.accuracy = OrderedDict()
        self.features_model = FaceRecognition(args)
        self.wrong_matches = []
        self.wrong_not_matches = []
        self.save_one_in = args.save_one_in
        print('Threshold - {}'.format(self.threshold))
        
    def processing_dataset(self, name_all_images):
        idx_images = 1
        for name in name_all_images:
            start = time.time()
            label = name[1]
            img = cv2.imread(name[0])
            #img = cv2.resize(img,(600,600))
            ### get features from image
            
            features,aligned = self.features_model.predict(img,1)
            if np.any(features):
                self._compare_feat(features, label, name)
                if idx_images%self.step_save_accuracy==0:
                    self._calculate_accuracy(idx_images)
                idx_images+=1
                #cv2.imwrite('Trash/{}'.format(name[0].split('/')[-1]),aligned[::,::,::-1])
            #if idx_images == 100:
             #   break
            #print('Processing one frame - {}'.format(time.time() - start))
        self._calculate_accuracy(idx_images)
        #return self.accuracy, self.wrong_matches, self.wrong_not_matches

    def _compare_feat(self, features, label, name):
        id_to_min = OrderedDict({'id': None, 'min': 100000})
        for key, val in self.id_to_feat.items():
            dist = distance.euclidean(val[0], features)
            #dist = np.linalg.norm(features - val[0], axis=1) 
            #print(dist)
            if dist < id_to_min['min']:
                id_to_min['id'] = key
                id_to_min['min'] = dist
        for feat_wrong in self.wrong_list:
            dist = distance.euclidean(feat_wrong[0], features)
            if dist < id_to_min['min']:
                id_to_min['id'] = None
                id_to_min['min'] = dist
        if id_to_min['min'] <= self.threshold:
            if label==id_to_min['id']:
                self.tr_pos+=1
            else:
                self.f_pos+=1
                if id_to_min['id']==None:
                    self.wrong_matches.append([name,None,id_to_min['min']])
                else:
                    self.wrong_matches.append([name,self.id_to_feat[id_to_min['id']][1],id_to_min['min']])
        else:
            if label not in self.id_to_feat:
                self.tr_neg+=1
                self.id_to_feat[label] = [features,name]
            else:
                self.f_neg+=1
                self.wrong_not_matches.append([name,self.id_to_feat[label][1],id_to_min['min']])
                self.wrong_list.append([features,name,id_to_min['min']])
    def _calculate_accuracy(self, idx_images):
        accuracy = (self.tr_neg+self.tr_pos)/(self.tr_pos+self.tr_neg+self.f_neg+self.f_pos)
        print('iteration - {}, accuracy -{}, fp - {}, fn - {}'.format(idx_images, accuracy, self.f_pos, self.f_neg))
        self.accuracy[idx_images] = accuracy
        if idx_images%self.save_one_in==0:
            with open('tf_1d_result.json', 'w') as fp:
                json.dump(self.accuracy, fp)
            np.save('wrong_matches.npy',self.wrong_matches)
            np.save('wrong_not_matches.npy',self.wrong_not_matches)
