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
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        
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
import cv2
import os
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread("pp.jpg",cv2.IMREAD_COLOR)
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
        

#%%









# %%

from tensorflow  import keras,data,numpy_function
from keras.layers import Conv2D,Concatenate,MaxPool2D,GlobalMaxPool2D,Dense,Dropout
from collections import defaultdict
import numpy as np
import pandas as pd
from keras.models import Model
from keras_vggface.vggface import VGGFace
from keras.metrics import Precision,Recall
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.utils import shuffle
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,TensorBoard,CSVLogger,EarlyStopping
from keras.optimizers import Adam


csv_path = os.path.join("metrics","metrics.csv")
model_path = os.path.join("metrics","senet.h5")
H = 224
W = 224


def shuffle1(X,Y):
    X,Y = shuffle(X,Y,random_state=42)
    return X,Y


def build_model(classes):
    
    senet = VGGFace(model='vgg16',include_top=False)
    senet.trainable = False
    
    inputs = senet.layers[-1]
    layer1 = GlobalMaxPool2D()(inputs.output)
    layer2 = Dense(512,activation='relu')(layer1)
    drop = Dropout(0.5)(layer2)
    output = Dense(classes,activation='softmax')(drop)
    Custom_Senet = Model(senet.input,output) 
    return Custom_Senet

    
    
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
        for pics in os.listdir(os.path.join(IMAGES,dirs)):
            dataset["images"] += [pics]
            dataset["Label"] += [dirs]

    df = pd.DataFrame(dataset)

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
         
        TRAIN_FINAL += [os.path.join("Train",map_labels[i],j) for j in TRAIN_TEMP["images"].values.tolist()]
        VALID_FINAL += [os.path.join("Train",map_labels[i],j) for j in VALID_TEMP["images"].values.tolist()]
        TRAIN_LABEL_FINAL+=to_categorical(TRAIN_TEMP["Label"],num_classes=len(map_labels)).tolist()
        VALID_LABEL_FINAL+=to_categorical(VALID_TEMP["Label"],num_classes=len(map_labels)).tolist()

    print(len(TRAIN_FINAL)+len(VALID_FINAL))
    print(len(TRAIN_LABEL_FINAL)+len(VALID_LABEL_FINAL))
    return (TRAIN_FINAL,TRAIN_LABEL_FINAL),(VALID_FINAL,VALID_LABEL_FINAL),len(map_labels)


def load_data(image,label):
    
    def _parse(image,label):
        img = load_image(image)
        return img,label
    
    img,label = numpy_function(_parse,[image,label],[np.float32,np.float32])
    return img,label

def tf_dataset(X,Y):
    dataset = data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.map(load_data)
    dataset = dataset.batch(24)
    dataset = dataset.prefetch(24)
    return dataset


    
if __name__ == "__main__":
    
    (TRAIN,TRAIN_LABEL),(VALID,VALID_LABEL),classes = load_dataset()
   
    TRAIN,TRAIN_LABEL = shuffle1(TRAIN, TRAIN_LABEL)
    TRAIN_DATASET = tf_dataset(TRAIN,TRAIN_LABEL)
    VALID_DATASET = tf_dataset(VALID,VALID_LABEL)
    Custom_Senet = build_model(classes)
    metrics = [Recall(),Precision()]
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]
    lr = 1e-4

    Custom_Senet.compile(loss="categorical_crossentropy",optimizer=Adam(lr),metrics = metrics)

    Custom_Senet.fit(
        TRAIN_DATASET,
        epochs=20,
        validation_data=VALID_DATASET,
        steps_per_epoch=len(TRAIN_DATASET),
        validation_steps=len(VALID_DATASET),
        callbacks=callbacks
    )
    
    
    
    
    
#%%

import os
import cv2
import numpy as np
from keras.models import load_model
Custom_Senet = load_model(os.path.join("metrics","senet(2).h5"))
img = cv2.imread('images.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img,(224,224))
img = np.expand_dims(img,axis=0)
res = Custom_Senet.predict(img)

print(np.argmax(res))



#%%
import cv2
from keras.models import load_model
import os
import numpy as np
model = load_model(os.path.join("metrics","senet(2).h5"))
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while(cap.isOpened):
    res , frame = cap.read()
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = face_classifier.detectMultiScale(rgb,minNeighbors=5,scaleFactor=1.1)
    
    for x,y,w,h in img:
        res = frame[y:y+h,x:x+w]
        res = cv2.resize(res,(224,224))
        res = np.expand_dims(res, axis=0)
        res = model.predict(res)
        
        stroke = 2
        color = (255,0,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
    
    cv2.imshow("Face Recognition",frame)
    if(cv2.waitKey(1) & 0XFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    





