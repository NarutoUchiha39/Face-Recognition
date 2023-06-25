#%%
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

Face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
Face_Recognizer = cv2.face.LBPHFaceRecognizer_create()

def open_image(image):
    
    img = cv2.imread(image)
    GRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = Face_classifier.detectMultiScale(GRAY,minNeighbors=5,scaleFactor=1.1)
    print(faces,image)
    if (len(faces) == 0):
        return None, None
    x,y,w,h = faces[0]
    rect = faces[0]
    return GRAY[y:y+h,x:x+w],rect

def draw_rect(image,x,y,w,h):
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    
def put_text(image,text,x,y):
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1)
    

def preprocess_data():
    PATH = "Train"
    TRAIN = []
    LABELS = []
    
    for labels in os.listdir(PATH):
            IMAGE_PATH = os.path.join(PATH,labels)
            folders = int(labels.replace("s",""))
            for images in os.listdir(IMAGE_PATH):
                image,rect = open_image(os.path.join(IMAGE_PATH,images))
                if(image is not None):
                    TRAIN.append(image)
                    LABELS.append(folders)
                    
    return TRAIN,np.array(LABELS)

def predict(image):
    image1,rect = open_image(image)
    image = cv2.imread(image,cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res = Face_Recognizer.predict(image1)
    draw_rect(image, rect[0], rect[1], rect[2], rect[3])
    put_text(image,res,rect[0], rect[1]-5)
    plt.plot(image)
    predict(res)
    
(TRAIN,LABELS) = preprocess_data()
print(LABELS)
Face_Recognizer.train(TRAIN,LABELS)
    
# %%
import cv2
import matplotlib.pyplot as plt
def open_image(image):
    
    img = cv2.imread(image)
    GRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = Face_classifier.detectMultiScale(GRAY,minNeighbors=5,scaleFactor=1.1)
    if (len(faces) == 0):
        return None, None
    x,y,w,h = faces[0]
    rect = faces[0]
    draw_rect(img, x, y, w, h)
    return GRAY[y:y+h,x:x+w],rect


def predict(image):
    image1,rect = open_image(image)
    image = cv2.imread(image)
    

    res = Face_Recognizer.predict(image1)
   
    draw_rect(image, rect[0], rect[1], rect[2], rect[3])
   
    put_text(image,str(res[0]),rect[0], rect[1]-5)
    plt.imshow(image)
    print(res)
    



predict(os.path.join("2829.jpeg"))


# %%
