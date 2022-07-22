import argparse
import cv2
import time
import numpy as np
import os
import sys
import copy
import torch
import random
import config
import json

from src.dataset.dataloader import img_to_tensor, uv_map_to_tensor
from src.dataset.dataloader import make_data_loader, make_dataset, ImageData
from src.model.loss import *
from PIL import Image
from src.util.printer import DecayVarPrinter
from src.visualize.render_mesh import render_face_orthographic, render_uvm
from src.visualize.plot_verts import plot_kpt, compare_kpt
from src.dataset.uv_face import uvm2mesh

from src.run.predict import SADRNv2Predictor

from glue import ULFace
from utils.landmark_utils import *
from utils.face_detection_utils import *

def predict(predictor, img):
	predictor.model.eval()
	resized = cv2.resize(img, dsize=(256,256))
	image = (resized / 255.0).astype(np.float32)
	for ii in range(3):
		image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
			image[:, :, ii].var() + 0.001)
	#cv2.imshow('1', image)
	image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
	with torch.no_grad():
		out = predictor.model({'img': image}, 'predict')
	
	#out['face_uvm'] *= config.POSMAP_FIX_RATE
	#out['kpt_uvm'] *= config.POSMAP_FIX_RATE

	#out['face_uvm'] = out['face_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
	#out['kpt_uvm'] = out['kpt_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
	out['offset_uvm'] = out['offset_uvm'].cpu().permute(0, 2, 3, 1).numpy()[0]
	out['attention_mask'] = out['attention_mask'].cpu().permute(0, 2, 3, 1).numpy()[0]	
		
	return out
	
def bbox_area(bbox):
	if bbox==None:
		return 0
	else:
		return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

def bbox_intersect(bbox_current, bbox_prev):
	if (bbox_current[0] > bbox_prev[2] or bbox_prev[0] > bbox_current[2]) \
		or (bbox_current[1] > bbox_prev[3] or bbox_prev[1] > bbox_current[3]):
		return None
	intersection[0] = max(bbox_current[0], bbox_prev[0])
	intersection[1] = max(bbox_current[1], bbox_prev[1])
	intersection[2] = min(bbox_current[2], bbox_prev[2])
	intersection[3] = min(bbox_current[3], bbox_prev[3])
	return intersection

# from glue.PFLD_TFLite import *

#sys.path.insert(1, '/glue')
#import glue

predictor_1 = SADRNv2Predictor('./data/saved_model/net_021.pth')
face_detector = ULFace.Detector()

cap = cv2.VideoCapture(0)

landmarks = np.empty( shape=(0, 0) )
bbox = None
bbox_prev = None
last_detection = None
is_face_detected = False

i = 0

xa, ya = 0, 0

while True:
	#print("\n\n-------------------------------------------------\n\n")
	ret, frame = cap.read()
	#print(frame)
	if not ret: break
	
	height, width = frame.shape[:2]
	
	frame_crop = None
	result = None
	intersection = [None]*4
	
	#is_face_detected, bbox = face_detector.detect_bbox(frame)
	
	is_landmarks_detected = landmarks.size != 0
	
	if (i == 0) or (i%20 == 0):
		is_face_detected, last_detection = face_detector.detect_bbox(frame)
		if is_face_detected and (not is_landmarks_detected):
			bbox = last_detection.copy()
			bbox_prev = last_detection
	if (i != 0) and is_face_detected and is_landmarks_detected:
		landmark_bbox = bbox_from_landmark(landmarks)
		
		last_detection_area = bbox_area(last_detection)
		intersect_area = bbox_area(bbox_intersect(last_detection, landmark_bbox))
		intersect_proportion = intersect_area/last_detection_area
			
		# print(intersect_proportion)
			
		if (intersect_proportion<0.5):
			is_face_detected, last_detection = face_detector.detect_bbox(frame)
			if is_face_detected:
				bbox = last_detection.copy()
				bbox_prev = last_detection
		else:
			bbox = bboxes_average(landmark_bbox, bbox_prev)
			bbox_prev = last_detection
		
	#landmarks = landmark_predictor.post_process(landmarks, bbox)
	
	if is_face_detected:
		cv2.rectangle(frame, (int(last_detection[0]), int(last_detection[1])),
				(int(last_detection[2]), int(last_detection[3])), (0, 0, 255), 2)
	
		bbox = face_detector.post_process(bbox)
	
		xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
		
		img = crop(frame, bbox)
	
		cv2.rectangle(frame, (xmin, ymin),
				(xmax, ymax), (125, 255, 0), 2)
	
		#start_time = time.perf_counter()
		out = predict(predictor_1, img)
		#print(time.perf_counter() - start_time)
		#print(landmarks)
	
		face_uvm_out = out['kpt_uvm'][0].cpu().permute(1, 2, 0).numpy() * (bbox[2]-bbox[0])*1.106
		#print(bbox[2]-bbox[0])
		output = face_uvm_out[uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
		#print(output)
	
		landmarks = np.array([(item[0]+bbox[0], item[1]+bbox[1]) for item in output])
	
		for (x, y) in landmarks:
				cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (0, 0, 255))
				
	else:
		landmarks = np.empty( shape=(0, 0) )
			
	cv2.imshow('1', frame)
	
	i = i+1
	
	if cv2.waitKey(1) == 27:
		break

	
cap.release()
cv2.destroyAllWindows()
