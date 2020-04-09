import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2D, Dense, Flatten, BatchNormalization

class Recognition(object):
    def __init__(self):
        self.model = self.build_model()


    def build_model(self):
        recognition_model = Sequential([
        Conv2D(8, 3, padding='same', activation='relu', input_shape=(80, 80 ,1)),
        MaxPooling2D(),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Dropout(.4),
        Flatten(),
        Dense(256, activation='relu'), #maybe not 512
        Dense(1, activation='sigmoid')
        ])

        recognition_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        return recognition_model

    def train_and_visualize(self, train_datagen, test_datagen, epochs = 50):
        recognition_model = self.model
        total_train, total_val, batch_size = 198, 52, 16
        history = recognition_model.fit_generator(
            train_datagen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=test_datagen,
            validation_steps=total_val // batch_size
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy (CNN)')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss (CNN)')
        plt.savefig('visualizations/cnn_stats.png')

