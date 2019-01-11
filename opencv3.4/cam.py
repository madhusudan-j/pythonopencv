# import urllib
# import cv2
# import numpy as np
# # print(cv2.getBuildInformation())
# url = "http://192.168.1.26:8080/shot.jpg"
# while True:
#     video = cv2.VideoCapture(url)
#     imgResp = urllib.urlopen(url)
#     imgNp = np.array(bytearray(imgResp.read()), dtype = np.uint8)
#     img = cv2.imdecode(imgNp, -1)
#     cv2.imshow('test', img)
#     if ord('q') == cv2.waitKey(10):
#         exit(0)

import numpy as np
import cv2

detector= cv2.CascadeClassifier('/home/comx-admin/ipcampro/home_surveillance/system/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0) # to stream video from laptop camera
# cap = cv2.VideoCapture("http://192.168.1.26:8080/video") # to stream video from mobilecam
cap = "http:admin:admin@//192.168.0.103:5000/onvif/device_service" # to stream video from ip camera http protocal 
# cap = "rtsp://192.168.0.103:554/onvif1" # to stream video from ip camera rtsp protocal
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()