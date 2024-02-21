## IMAGE CLASSIFICATION ##
# Author: Bethany Schoen
# Date: 21/02/2024
##########################
# This python file explores how to use Keras to construct a convolutional neural network to classify chest x rays

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report

## VARIABLES ##
# data downloaded from: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
path_to_image = r"C:\Users\Beth\Documents\Python Scripts\Codecademy\image_classification_data\Covid19-dataset\train\Covid\COVID-00017.jpg"
training_data_loc = r'C:\Users\Beth\Documents\Python Scripts\Codecademy\image_classification_data\Covid19-dataset\train'
validation_data_loc = r'C:\Users\Beth\Documents\Python Scripts\Codecademy\image_classification_data\Covid19-dataset\test'
batch_size = 32
num_classes = 3

def get_image_dimensions(path_to_image: str):
    """
    Find out data dimensions
    """
    img = Image.open(path_to_image)
    width, height = img.size

    return (width, height)

def prepare_training_and_validation_data(img_dimensions: tuple, training_data_loc: str, validation_data_loc: str, batch_size: int):

    training_data_generator = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    training_iterator = training_data_generator.flow_from_directory(
        training_data_loc, 
        class_mode='categorical', 
        color_mode='grayscale', 
        target_size=img_dimensions, 
        batch_size=batch_size)

    validation_data_generator = ImageDataGenerator(
        rescale=1/255.0)

    validation_iterator = validation_data_generator.flow_from_directory(
        validation_data_loc, 
        class_mode='categorical', 
        color_mode='grayscale', 
        target_size=img_dimensions, 
        batch_size=batch_size)

    return training_iterator, validation_iterator


# model
def custom_create_model(kernal_size=(3, 3), 
                pool_size=(2, 2), 
                num_neurons=16, 
                num_filters=8, 
                strides=2, 
                padding='valid', 
                learning_rate=0.01):
  """
  Model for image classification with customisable hyperparameters
  """
  model = Sequential()
  input_layer = InputLayer(input_shape=(256, 256, 1))
  model.add(input_layer)
  # alternatively, can specify input shape in first Conv2D layer to automatically add an input (according to ChatGPT):
  # Add a convolutional layer with 32 filters, each with a 3x3 kernel, and ReLU activation function
  #model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 1)))
  model.add(Conv2D(num_filters, kernal_size, strides=strides, activation='relu', padding=padding))
  model.add(MaxPooling2D(pool_size, strides=strides, padding=padding))
  model.add(Conv2D(num_filters, kernal_size, strides=strides, activation='relu', padding=padding))
  model.add(MaxPooling2D(pool_size, strides=strides, padding=padding))
  model.add(Flatten())
  model.add(Dense(num_neurons, activation='relu'))

  output_layer = Dense(3, activation='softmax')
  model.add(output_layer)

  opt = Adam(learning_rate=learning_rate)
  # same as calling categorical_crossentropy immediately in the model.compile function
  # this way gives more flexibility - can customise loss function
  loss = CategoricalCrossentropy()
  model.compile(loss=loss, optimizer=opt, metrics=[CategoricalAccuracy(),AUC()])

  return model

# AlexNet model
# Source: https://www.kaggle.com/code/blurredmachine/alexnet-architecture-a-complete-guide
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.add(InputLayer(input_shape=(input_shape[0], input_shape[1], 1)))

        self.add(Conv2D(96, kernel_size=11, strides= 4,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        self.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)) 

        self.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        self.add(Flatten())
        self.add(Dense(4096, activation= 'relu'))
        self.add(Dense(4096, activation= 'relu'))
        self.add(Dense(1000, activation= 'relu'))
        self.add(Dense(num_classes, activation= 'softmax'))

        self.compile(optimizer= Adam(0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

def train_model(model, training_iterator, validation_iterator):

    stop = EarlyStopping(monitor='val_loss', mode='min', patience=40)
    history = model.fit(training_iterator, 
        steps_per_epoch=training_iterator.samples/batch_size,
        epochs=100,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples/batch_size,
        callbacks=[stop]
    )

    return model, history

def visualise_training_history(history):
    """
    Having trained the model, plot how the validation loss and accuracy changed with model updates
    """
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('static/images/my_plot1.png')

    # Plot training and validation accuracy
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/images/my_plot1.png')

    return

def evaluate_model():
    """
    Having trained the model, evaluate it's performance using the validation data
    """
    test_steps_per_epoch = np.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
    predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_iterator.classes
    class_labels = list(validation_iterator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report) 

    return

if __name__ == "__main__":
    img_dimensions = get_image_dimensions(path_to_image)
    training_iterator, validation_iterator = prepare_training_and_validation_data(img_dimensions, training_data_loc, validation_data_loc, batch_size)
    basic_model = custom_create_model()
    alex_model = AlexNet(img_dimensions, num_classes)
    model, history = train_model(alex_model, training_iterator, validation_iterator)
    visualise_training_history(model)

