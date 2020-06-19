# -*- coding: utf-8 -*-
"""
Created on Sat May 30 03:49:51 2020

@author: rajan
"""

# -*- coding: utf-8 -*-

import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
             
        blur = cv2.GaussianBlur(roi_color, (51,51), 0)
        frame[y:y+h,x:x+w]=blur

    cv2.imshow("frame",frame)
    k= cv2.waitKey(30) & 0xff   #press escape to exit
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()