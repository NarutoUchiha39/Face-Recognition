#%%
from collections import defaultdict
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#link = "https://192.168.29.171:8080/video"
cap = cv2.VideoCapture(0)
#cap.open(link)
while(cap.isOpened):
    success,frame = cap.read()    
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(rgb,minNeighbors=5,scaleFactor=1.1)
    
    for x,y,w,h in faces:
        stroke = 2
        color = (0,256,256)
        cv2.rectangle(frame,(x,y),(x+w),(y+h),color,stroke)
        
    cv2.imshow("Face Recognition",frame)
    if(cv2.waitKey(1) & 0XFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()
        
#%%
from keras_vggface.vggface import VGGFace
import keras.utils as image
from keras_vggface import utils
import numpy
import cv2

Face_Detection = VGGFace(model='senet50')
img = cv2.imread("obama.jpeg",cv2.IMREAD_COLOR)
img = cv2.resize(img,(224,224))
img = img.astype(numpy.float64)
img = numpy.expand_dims(img,axis=0)
img = utils.preprocess_input(img,version=1)

res = Face_Detection.predict(img)
print(utils.decode_predictions(res))




# %%
import matplotlib.pyplot as plt
import numpy
img = cv2.imread("obama.jpeg",cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_array = numpy.array(img,numpy.uint8)
predict = face_classifier.detectMultiScale(rgb,minNeighbors=5,scaleFactor=1.1)
for x,y,w,h in predict:

    face = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face)
plt.show()
    
#%%
import os
PATH = "lfw"
NEW_PATH = "Train"
for dirs in os.listdir(PATH):
    length = len(os.listdir(os.path.join(PATH,dirs)))
    
    if(length>100):
        os.replace(os.path.join(PATH,dirs),os.path.join(NEW_PATH,dirs))
        




# %%

from tensorflow  import keras,data
from keras.layers import Conv2D,Concatenate,MaxPool2D
import numpy as np
import pandas as pd
from keras_vggface.vggface import VGGFace
from keras.metrics import Precision,Accuracy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from collections import defaultdict

H = 224
W = 224

def load_image(Path):
    Path = Path.decode()
    img = cv2.imread(Path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(H,W))
    img = img/255.0
    img = img.astype(np.float32)
    return img


def load_dataset():
    IMAGES = "Train"
    dataset  = defaultdict(list)
    for dirs in os.listdir(IMAGES):
        images = []
        for pics in os.listdir(os.path.join(IMAGES,dirs)):
            dataset["images"] += [pics]
            dataset["Label"] += [dirs]

    df = pd.DataFrame(dataset)
    grouped_data = df.groupby(by="Label")

    Encoder = LabelEncoder()
    Labels = Encoder.fit_transform(df["Label"])
    df = df.drop(['Label'],axis=1)
    df["Label"] = Labels

    map_labels = {index:label for index,label in enumerate(Encoder.classes_)}
    print(map_labels)

    TRAIN_FINAL = []
    VALID_FINAL = []

    TRAIN_LABEL_FINAL = []
    VALID_LABEL_FINAL = []

    

    for i in map_labels:

        TRAIN_TEMP = []
        VALID_TEMP = []
        
        filt = df["Label"] == i
        df_temp = df[filt] 
        
        TRAIN_TEMP,VALID_TEMP= train_test_split(df_temp,test_size = 0.2,random_state=42)
        TRAIN_FINAL += TRAIN_TEMP["images"].values.tolist()
        VALID_FINAL += VALID_TEMP["images"].values.tolist()
        TRAIN_LABEL_FINAL+=to_categorical(TRAIN_TEMP["Label"],num_classes=len(map_labels)).tolist()
        VALID_LABEL_FINAL+=to_categorical(VALID_TEMP["Label"],num_classes=len(map_labels)).tolist()

    print(len(TRAIN_FINAL)+len(VALID_FINAL))
    print(len(TRAIN_LABEL_FINAL)+len(VALID_LABEL_FINAL))


load_dataset()
        

        



    








# %%
