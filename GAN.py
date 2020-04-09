import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
import time
import os
import numpy as np
from data import GeneratedImg


class GAN(object):
    def __init__(self):
        self.artist = self.build_artist()
        self.critic = self.build_critic()
        self.artist_opt = tf.keras.optimizers.Adam(1e-4)
        self.critic_opt = tf.keras.optimizers.Adam(1e-4)

    def build_artist(self):
        model = Sequential([
            Dense((20 * 20 * 80), use_bias=False, input_shape=(100,)),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((20, 20, 80)),
            Conv2DTranspose(128, (4, 4), use_bias=False, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (4, 4), strides=(2, 2), use_bias=False, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(1, (4, 4), strides=(2, 2), use_bias=False, padding='same')
        ])
        return model

    def build_critic(self):
        model = Sequential([
            Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[80, 80, 1]),
            LeakyReLU(),
            MaxPooling2D(),
            Dropout(.5),
            Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            LeakyReLU(),
            Dropout(.3),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1)
        ])
        return model

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_step(self, images, BATCH_SIZE=16):
        noise_dim = 100

        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.artist(noise, training=True)
            real_output = self.critic(images, training=True)
            fake_output = self.critic(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.artist.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.critic.trainable_variables)

        self.artist_opt.apply_gradients(zip(gradients_of_generator, self.artist.trainable_variables))
        self.critic_opt.apply_gradients(zip(gradients_of_discriminator, self.critic.trainable_variables))

    def generate_and_save_images(self, epoch, test_input, save_file):
        generated_imgs = self.artist(test_input, training=False)
        confs = self.critic(generated_imgs, training=False)
        for num, (conf, img) in enumerate(zip(confs, generated_imgs)):
            temp_id = str(epoch) + "_" + str(num)
            file = GeneratedImg(img, epoch, conf, temp_id)
            file.save(f)

    def train(self, dataset, epochs, save_file):
        seed = tf.random.normal([16, 100])
        with open(save_file, 'w') as f:
            for epoch in range(epochs):
                start = time.time()
                for image_batch in dataset:
                    self.train_step(image_batch)
                self.generate_and_save_images(epoch + 1, seed, f)
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            self.generate_and_save_images(epochs+1, seed, f)
