import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import to_categorical

img_width, img_height = 128, 128

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 17680
nb_validation_samples = 2192
epochs = 50
batch_size = 16

def createGenerators():
    global train_generator, val_generator
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    val_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

def saveBottlebeckFeatures():
    model = applications.VGG16(include_top=False, weights='imagenet')

    bottleneck_features_train = model.predict_generator(
        train_generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    
    bottleneck_features_validation = model.predict_generator(
        val_generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def trainTopModel():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = to_categorical(train_generator.classes, 20)

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels =to_categorical(val_generator.classes, 20)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(20, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

createGnerators()
saveBottlebeckFeatures()
trainTopModel()