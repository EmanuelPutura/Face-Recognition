import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FaceRecognitionPipeline:
    def __init__(self, train_data_dir, validation_data_dir):
        # Collect and pre-process the dataset
        self.__datagen = ImageDataGenerator(rescale=1. / 255)

        # Load the data from the directory structure
        self.__train_data = datagen.flow_from_directory(train_data_dir, target_size=(224, 224))
        self.__validation_data = datagen.flow_from_directory(validation_data_dir, target_size=(224, 224))

        self.__model = FaceRecognitionModel(train_data.num_classes)

        # Compile the model
        self.__model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=10):
        return self.__model.fit_generator(self.__train_data, epochs=epochs, validation_data=self.__validation_data)

    def evaluate_model(self, test_data_dir):
        test_data = datagen.flow_from_directory(test_data_dir, target_size=(224, 224))
        return model.evaluate_generator(test_data)

    # Use the model to match a person's face with their name
    def match_name(self, image):
        # Pre-process the image
        image = np.expand_dims(image, axis=0)
        image = self.__datagen.standardize(image)

        # Predict the name using the model
        predict = self.__model.predict(image)
        name_probs = predict[0]

        name_index = np.argmax(name_probs)
        probability = predict[0][name_index]

        return self.__train_data.class_indices[name_index], probability


class FaceRecognitionModel(tf.keras.Sequential):
    def __init__(self, num_classes):
        super().__init__()

        # Choose a model architecture
        self.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        self.add(keras.layers.MaxPooling2D((2, 2)))
        self.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(keras.layers.MaxPooling2D((2, 2)))
        self.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.add(keras.layers.MaxPooling2D((2, 2)))
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(128, activation='relu'))
        self.add(keras.layers.Dense(num_classes, activation='softmax'))
