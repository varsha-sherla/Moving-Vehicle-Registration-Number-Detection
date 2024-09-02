#Install Ultralytics for YOLOv8
!pip install ultralytics
#Install EasyOCR for extracting text from image
!pip install easyocr
#Install filterpy,scikit-image and lap for SORT tracker
!pip install filterpy
!pip install scikit-image
!pip install lap

import os
from ultralytics import YOLO
from sort import *
import cv2
import easyocr
import numpy as np

#initiating reader
reader = easyocr.Reader(['en'], gpu=True)

#initialize SORT tracker
mot_tracker=Sort()


def ocr_image(img,x,y,w,h):
    x,y,w,h=int(x),int(y),int(w),int(h)
    img = img[y:h,x:w]

    gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    result = reader.readtext(gray)
    text = ""
    conf=0
    for res in result:
        if len(result) == 1:
            text = res[1]
            conf=res[2]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
            conf=res[2]

    text=text.upper().replace(" ","")
    x=['(',')','{','}','[',']',',','.','-','_']
    for i in x:
      text=text.replace(i,"")

    return str(text),conf



def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1



def get_best(results,v):
  dcty={}
  lt=[]
  for k,v in results.items():

    for m,n in v.items():
      if m not in dcty:
        dcty[m]={'text': n['text'],
                'text_sc': n['text_score']}
      else:
        if dcty[m]['text_sc']<n['text_score']:
          dcty[m]={'text': n['text'],
                  'text_sc': n['text_score']}

  for i,j in dcty.items():
    if len(j['text'])>4:
      lt.append(j['text'] )
  lt=list(set(lt))
  return lt



if __name__ == "__main__":
  coco_model=YOLO('yolov8n.pt')
  lpdetect=YOLO('best_5.pt')
  cap=cv2.VideoCapture('kia_alto.mp4')
  
  results = {}
  vehicles = [2, 3, 5, 7]
  # read frames
  frame_nmr = -1
  ret = True
  while ret:
      frame_nmr += 1
      ret, frame = cap.read()
      if ret:
          # detect vehicles
          detections = coco_model(frame)[0]
          detections_ = []
          for detection in detections.boxes.data.tolist():
              x1, y1, x2, y2, score, class_id = detection
              if int(class_id) in vehicles:
                  detections_.append([x1, y1, x2, y2, score])

          # track vehicles
          track_ids = mot_tracker.update(np.asarray(detections_))
          # detect license plates
          license_plates = lpdetect(frame)[0]
          for license_plate in license_plates.boxes.data.tolist():
              x1, y1, x2, y2, score, class_idd = license_plate
              # assign license plate to car
              xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
              if car_id != -1 and class_idd!=-1:
                  results[frame_nmr] = {}
                  lplate_text,txt_score=ocr_image(frame,x1,y1,x2,y2)
                  if lplate_text is not None:
                      results[frame_nmr][car_id] = {'text': lplate_text,
                                                    'text_score': txt_score}

  number_plates=get_best(results,v)
  print(number_plates)


