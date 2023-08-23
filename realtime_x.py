from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from pygame import mixer
import numpy as np
import imutils
import time
import cv2
import os
from matplotlib import pyplot as plt
from threading import Timer
import time


deploy = r"face_detector\deploy.prototxt"                                  
face_detect = r"face_detector\face_detection_res_net.caffemodel"     
faceNet = cv2.dnn.readNet(deploy, face_detect)                             
maskNet = load_model("powerhouse_d.model") 


    
def face_mask_detection(frame, faceNet, maskNet):
    IMG_SIZE = 224
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            #face = face.reshape(-1,IMG_SIZE, IMG_SIZE, 3)
            face  =preprocess_input(face)
            #plt.imshow(face)
            #plt.show()
            #print(face)
        
            
            faces.append(face)

    if len(faces) > 0:
        #print(len(faces))
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces)


    return preds

     

vs = VideoStream(src=0).start()
mixer.init() 
mixer.music.load("Facemask Sound_1.mp3") #Loading Music File

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500, height = 500)
    #frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = face_mask_detection(frame, faceNet, maskNet)

   
    #print(len(preds))
    if (len(preds) != 0):
       for i in range (len(preds)):
           if (preds[i][0] > preds[i][1]): 
               label = "Mask" 
           else:
               label = "No Mask"
               
           if (label == "No Mask"):
               if mixer.music.get_busy() == False: mixer.music.play() 
       
        
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()