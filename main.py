from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import matplotlib.pyplot as plt

top_model_weights_path = 'bottleneck_fc_model.h5'
model_save_path = 'model_saved3.h5'

img_width, img_height = 128, 128

train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'

epochs = 80
batch_size = 16

def createGenerators():
    global train_generator, validation_generator

    train_datagen = ImageDataGenerator(
        rotation_range=45,
        rescale=1 /255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

def createModel():
    """input_tensor = Input(shape=(img_width,img_height,3))
    base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(20, activation='softmax'))

    top_model.load_weights(top_model_weights_path)

    model = Model(input= base_model.input, output= top_model(base_model.output))

    for layer in model.layers[:15]:
        layer.trainable = False"""

    base_model = load_model(model_save_path)
    ll = base_model.output
    ll = Dense(2,activation="softmax")(ll)

    model = Model(inputs=base_model.input,outputs=ll)

    return model

def trainModel(model):

    model.compile(loss='categorical_crossentropy',
        optimizer='adagrad',
        metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_steps = validation_generator.samples // batch_size,
        shuffle = True)

    model.summary()

def evaluateAndCreatePlot(model):
    ev = model.evaluate_generator(validation_generator)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('Accuracy: %.2f' % (ev[1]*100))

model = createModel()
createGenerators()
trainModel(model)
evaluateModelAndCreatePlot(model)

model.save(model_save_path)