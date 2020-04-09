
import sys
import os
import data
import recognition
import GAN
import tensorflow as tf


#main function
if __name__ =='__main__':
    """
    TODO:
    Add a pipeline so that the data generated will be automatically pushed into the 
    classifier's training data and it can be retrained in real time. Print results
    while on the vm.
    """
    print('Usage: python3 main.py homogeneous_data_to_be_replicated baseline_data file_save')
    print('Tensorflow Version: ', tf.__version__)
    imgs_to_generated = sys.argv[1:][0]
    base_line_imgs = sys.argv[1:][1]
    file = sys.argv[1:][2]
    gen_data = data.Data(imgs_to_generated, base_line_imgs).data_array
    gan_instance = GAN.GAN()
    gan_instance.train(gen_data, 300, file)




