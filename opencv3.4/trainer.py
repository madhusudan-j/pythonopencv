import os
import cv2
import numpy as np 
from PIL import Image
# print(help(cv2.face))
# recognizer = cv2.LBPHFaceRecognizer_create()
# recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'
def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print ID 
        Ids.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return Ids, faces

Ids, faces = getImageWithId(path)
recognizer.train(faces, np.array(Ids))
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()
