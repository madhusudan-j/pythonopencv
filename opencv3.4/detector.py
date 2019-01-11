import numpy as np
import cv2

detector= cv2.CascadeClassifier('/home/comx-admin/ipcampro/home_surveillance/system/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.43.1:8080/video") # to stream video from mobilecam

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingdata.yml")
id = 0 
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 4)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if id == 1:
            id = "madhu"
        elif id == 5:
            id = "gopal"
        elif id == 6:
            id = "sagar"
        else:
            id = "unknown"
        # cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x, y+h), font, 255)
        cv2.putText(img,str(id), (x,y+h),font, 2, 255)
        # img,(x,y),(x+w,y+h),(255,0,0),2
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()