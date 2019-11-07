from keras.preprocessing.image import ImageDataGenerator 
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

img_width, img_height= 128 , 128

loaded_model = load_model('model_saved3.h5')

validation_data_dir = 'v_data/test/' 
val_datagen = ImageDataGenerator(rescale=1. / 255) 
  
val_generator = val_datagen.flow_from_directory( 
    validation_data_dir) 

fig = plt.figure(figsize=(20, 20))

batch_holder = np.zeros((40, img_width, img_height, 3))
img_dir='prediction/images/'
for i,img in enumerate(os.listdir(img_dir)):
  img = image.load_img(os.path.join(img_dir,img), target_size=(img_width,img_height))
  batch_holder[i, :] = img

result = loaded_model.predict(batch_holder)
y_classes = result.argmax(axis=-1)

labelsDic = val_generator.class_indices
labels = list()
for key, value in labelsDic.items():
    labels.append(key)

labels = sorted(labels)

for i,img in enumerate(batch_holder):
  fig.add_subplot(4,14, i+1).axis('Off')
  plt.title(labels[y_classes[i]], fontsize=3)
  plt.imshow(img/256.)
  
plt.show()