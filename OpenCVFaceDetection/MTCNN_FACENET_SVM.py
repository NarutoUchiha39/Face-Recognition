#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras_facenet import FaceNet

Face_Classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_path = os.path.join("Facenet","facenet_keras.h5")
embedder = FaceNet()

def normalize(image):
    print(image,image.shape)
    print("=============")
    if(image.ndim == 4):
        axis = (1,2,3)
        size = image[0].size
        
    if(image.ndim == 3):
        axis = (0,1,2)
        size = image.size
    mean = np.mean(image,axis=axis,keepdims=True)
    deviation =np.maximum(np.std(image,axis=axis,keepdims=True),1.0/np.sqrt(size))
    gauss = (image-mean)/deviation
    return gauss

def L2_Normalize(x,epsilon = 1e-10,axis=-1):
    print(x)
    print("=============")
    x = x/np.sqrt(np.maximum(np.sum(np.square(x),axis=-1,keepdims=True),epsilon))
    return x

def preprocessing(path,margin=10):
    
    images = []
    for file in path:
        img = cv2.imread(file)
        #color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        res = Face_Classifier.detectMultiScale(img,scaleFactor = 1.1,minNeighbors=3)
        try:
            x,y,w,h = res[0]
            cropped_image = img[y:y+h,x:x+w]
            
            img = cv2.resize(cropped_image,(160,160))
            images.append(img)
        except Exception as E:
            continue
        
    return np.array(images)

def calc_embs(filepaths):
    aligned_images = normalize(preprocessing(filepaths))
    pd = []
    for start in range(0, len(aligned_images)):
        pd.append(embedder.embeddings(np.expand_dims(aligned_images[start],axis=0)))
    embs = L2_Normalize(np.concatenate(pd))

    return embs

def train(names):
    
    labels = []
    embs = []
    
    for i in names:
        path = os.path.join("Train",i)
        images_path = [os.path.join(path,files) for files in os.listdir(path)][:100]
        embs_ = calc_embs(images_path)
        labels.extend([i]*len(embs_))
        embs.append(embs_)
        
    embs = np.concatenate(embs)
    Encoder = LabelEncoder()
    labels = Encoder.fit_transform(labels)
    label_mapings ={index:labels for index,labels in enumerate(Encoder.classes_)}
    print(label_mapings)
    clf = SVC(kernel='linear', probability=True).fit(embs, labels)
    return Encoder, clf



def infer(le, clf, filepaths):
    embs = calc_embs(filepaths)
    print(embs)
    pred = le.inverse_transform(clf.predict(embs))
    return pred    

        
        
Encoder, Support_Vector_Machine = train(["s1","s2","s3","s4","s5"])
    
#%%

res = infer(Encoder, Support_Vector_Machine,[os.path.join("Train","s5","Tony_Blair_0004.jpg")])
print(res)












# %%
