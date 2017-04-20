# coding:utf-8

from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
from keras.datasets import mnist
from .MultiGPU import ToMultiGPU


class MnistClassification:

    def __init__(self):
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 20
        # Image size
        self.img_rows = 28
        self.img_cols = 28
        
        
    def model_definition(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model = ToMultiGPU.to_multi_gpu(self.model)

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])


    def read_dataset(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        if K.image_data_format() == 'channels_first':
            self.x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
            
        else:
            self.x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)
            
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)
        

    def __call__(self):
        self.read_dataset()
        self.model_definition()

        # Run
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test))

        self.score = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        print('Test loss:', self.score[0])
        print('Test accuracy:', self.score[1])
    

if __name__ == '__main__':
    test = MnistClassification()
    test()
