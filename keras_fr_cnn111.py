# python -m pip install h5py
from __future__ import print_function
from keras.optimizers import SGD
import os
import sys
from PIL import Image
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
import numpy as np
from keras.callbacks import TensorBoard
# for reproducibility  用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随即数都相同
np.random.seed(1337)


'''
Olivetti Faces是纽约大学的一个比较小的人脸库，由40个人的400张图片构成，即每个人的人脸图片为10张。每张图片的灰度级为8位，每个像素的灰度大小位于0-255之间。整张图片大小是1190 × 942，一共有20 × 20张照片。那么每张照片的大小就是（1190 / 20）× （942 / 20）= 57 × 47 。
'''

# There are 40 different classes
nb_classes = 80  # 40个类别
epochs = 60 # 进行40轮次训
batch_size = 40  # 每次迭代训练使用40个样本

# input image dimensions
img_rows, img_cols, img_deepth = 413, 579, 3
# number of convolutional filters to use
nb_filters1, nb_filters2 = 5, 10  # 卷积核的数目（即输出的维度）
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3  # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。

name= []

def load_data(rootdir):
    list = os.listdir(rootdir) 
    #print(list)
    faces = np.empty((nb_classes*10, 717381))
    label = np.empty(nb_classes*10)
    dict= {}
    #label = np.ndarray.astype(“str”) 
    n=0
    ii=-1
    for i in list:
        pathphoto = os.path.join(rootdir,i)
        namei= i[0:len(i)-6]
        if(not namei in dict):
            ii=ii+1
            dict[namei]=ii
            name.append(namei)
        img = Image.open(pathphoto)
        img_ndarray = np.asarray(img, dtype='float64')
        faces[n]=np.ndarray.flatten(img_ndarray)
        label[n]=ii
        n=n+1
    print("Load data successful!")
    label = label.astype(np.int)
    train_data = np.empty((nb_classes*8, 717381))
    train_label = np.empty(nb_classes*8)
    valid_data = np.empty((nb_classes, 717381))
    valid_label = np.empty(nb_classes)
    test_data = np.empty((nb_classes, 717381))
    test_label = np.empty(nb_classes)

    nn=0
    for i in range(16):
        for j in range(8):
            for k in range(5):
                train_data[nn]=faces[i*50+k*10+j]
                train_label[nn]=label[i*50+k*10+j]
                nn=nn+1

    for i in range(nb_classes):
        #train_data[i*8: i*8+8] = faces[i*10: i*10+8]  # 训练集中的数据
        #train_label[i*8: i*8+8] = label[i*10: i*10+8]  # 训练集对应的标签
        valid_data[i] = faces[i*10+8]  # 验证集中的数据
        valid_label[i] = label[i*10+8]  # 验证集对应的标签
        test_data[i] = faces[i*10+9]
        test_label[i] = label[i*10+9] 

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')

    rval = [(train_data, train_label), (valid_data,
                                        valid_label), (test_data, test_label)]
    return rval


def set_model(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(15, kernel_size=(3, 3),
                         input_shape=(img_deepth, img_rows, img_cols)))
    else:
        model.add(Conv2D(15, kernel_size=(3, 3),
                         input_shape=(img_rows, img_cols, img_deepth)))
    model.add(Activation('sigmoid'))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Conv2D(15, kernel_size=(3, 3)))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))  # Full connection
    model.add(Activation('tanh'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_val, Y_val),callbacks=[TensorBoard(log_dir='./log')])
    model.save_weights('model_weights.h5', overwrite=True)
    return model


def test_model(model, X, Y):
    model.load_weights('model_weights.h5')
    score = model.evaluate(X, Y, verbose=0)
    return score


if __name__ == '__main__':
    # the data, shuffled and split between tran and test sets
    
    rootdir='./dataset'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(rootdir)

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], img_deepth, img_rows, img_cols)
        X_val = X_val.reshape(X_val.shape[0], img_deepth, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_deepth, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_deepth)
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, img_deepth)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_deepth)
        input_shape = (img_rows, img_cols, img_deepth)  # 1 为图像像素深度

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = set_model()
    train_model(model, X_train, Y_train, X_val, Y_val)
    score = test_model(model, X_test, Y_test)
    
    model.load_weights('model_weights.h5')
    classes = model.predict_classes(X_test, verbose=0)
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("accuarcy:", test_accuracy)
    for i in range(0, nb_classes):
        if y_test[i] != classes[i]:
            print(name[int(y_test[i])], '被错误分成', name[int(classes[i])])