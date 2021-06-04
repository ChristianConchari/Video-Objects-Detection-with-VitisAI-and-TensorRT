#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, time 
import numpy as np
import cv2
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

model = load_model('models/numbers_model.h5')

#prev_img=cv2.imread('/home/ubuntu/tensorrt/URFD_images/Falls/fall_fall-02/frame0039.jpg')
#curr_img=cv2.imread('/home/ubuntu/tensorrt/URFD_images/Falls/fall_fall-02/frame0040.jpg')


cap = cv2.VideoCapture('numbers_video.mp4')
#cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prev_img=frame
prev_img_res=cv2.resize(prev_img,(32,32))
prev_img_c=cv2.cvtColor(prev_img_res, cv2.COLOR_BGR2GRAY)
kernel = np.ones((15,15),np.float32)/225
counter=0
acc=0.0

while(True):
	ret, frame = cap.read()
	curr_img_res=cv2.resize(frame,(32,32)) 

	fall = np.expand_dims(curr_img_res ,axis=0)
	start_time = time.time()
	fall_norm = fall/255.0
	preds=model.predict(fall_norm)
	#acc=acc+(time.time()-start_time)
	if preds[0][0] < 0.5:
		print(preds, 'Detected: 2'," FPS: ", 1.0/(time.time()-start_time))
		cv2.putText(frame, 
                'Two',
                (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                (255, 0, 0), 
                2, 
                cv2.LINE_AA)
	else:
		print(preds, 'Detected: 0'," FPS: ", 1.0/(time.time()-start_time))
		cv2.putText(frame, 
                'Zero',
                (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA)
	
	cv2.imshow("frame", frame)
	cv2.waitKey(1)
	acc=acc+1.0/(time.time()-start_time)
	counter=counter+1
	if counter>=2250:
		print('average FPS:',acc/counter)
		break