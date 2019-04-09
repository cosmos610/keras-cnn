import os
import numpy as np
from PIL import Image
nb_classes=92
rootdir="./dataset"
list = os.listdir(rootdir) 
faces = np.empty((nb_classes*10, 717381))
label = np.empty(nb_classes*10)
dict= {}
name=[]
#label = np.ndarray.astype(“str”) 
n=0
ii=-1
for i in list:
    pathphoto = os.path.join(rootdir,i)
    namei= i[0:len(i)-6]
    if(not namei in dict):
        ii=ii+1
        print(ii,namei)
        dict[namei]=ii
        name.append(namei)
    img = Image.open(pathphoto)
    img_ndarray = np.asarray(img, dtype='float64')
    faces[n]=np.ndarray.flatten(img_ndarray)
    label[n]=ii
    n=n+1