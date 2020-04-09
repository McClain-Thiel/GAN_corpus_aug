import tensorflow as tf
import numpy as np
import sys
import os
import json


from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

class Data(object):
    def __init__(self, gen_directory, rec_directory):
        print(gen_directory, rec_directory)
        self.target_size = (80, 80)
        self.batch_size = 16
        self.train_datagen = self.build_train_datagen(rec_directory)
        self.test_datagen = self.build_test_datagen(rec_directory)
        self.data_array = self.build_gen_array(gen_directory)
        self.original_training_array = ...
        self.original_val_array = ...



    def build_train_datagen(self, directory):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        train_gen = train_datagen.flow_from_directory(directory + '/training',
                                                      target_size=self.target_size, color_mode='grayscale',
                                                      batch_size=self.batch_size,
                                                      class_mode='binary')
        return train_gen

    def build_test_datagen(self, directory):
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        val_gen = test_datagen.flow_from_directory(directory + '/val',
                                                   target_size=self.target_size, color_mode='grayscale',
                                                   batch_size=self.batch_size,
                                                   class_mode='binary')

        return val_gen


    def build_gen_array(self, directory, num_imgs=133, batch_size=10):
        print(directory)
        class_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        class_gen = class_datagen.flow_from_directory(directory,
                                                      target_size=self.target_size, color_mode='grayscale',
                                                      batch_size=batch_size,
                                                      class_mode='input')
        arr = np.array(next(class_gen)[0])
        for x in range(int(num_imgs/batch_size) - 1):
            arr = np.append(arr, next(class_gen)[0], axis=0)

        print(arr.shape)

        arr = arr.reshape((num_imgs-3, 1, self.target_size[0], self.target_size[1], 1))
        return arr

class GeneratedImg(object):
    def __init__(self, img, epoch, confidence, id_sting):
        self.epoch = epoch
        self.img = img
        self.confidence = confidence
        self.ID = id_sting


    def save(self, file):
        json.dump({
            'Picture': self.img.encode('base64'),
            'Epoch': self.epoch,
            'Confidence': self.confidence
            }, file)



