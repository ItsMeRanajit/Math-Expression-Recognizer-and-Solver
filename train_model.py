# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %% [markdown]
# Train the model

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# %%
def binarize(img):
    img = image.img_to_array(img, dtype='uint8')
    binarized = np.expand_dims(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2), -1)
    inverted_binary_img = ~binarized
    return inverted_binary_img

# %%
data_dir = 'archive/data/extracted_new_images2'
batch_size = 32
img_height = 45
img_width = 45

# %%
train_datagen = ImageDataGenerator(
    preprocessing_function=binarize)

# %%
train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        seed=123)

# %%
# Class names
class_names = [k for k,v in train_generator.class_indices.items()]
class_names
print(class_names)

# %%
num_classes = 23

model = tf.keras.Sequential([
  tf.keras.layers.Input((45, 45, 1)),
  tf.keras.layers.Rescaling(1./255), 
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# %%
model.summary()

# %%
model.compile(
  optimizer='adam',
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# %%
model.fit(
  train_generator,
  epochs=5
)

# %%
model.save('eqn-detect-model2.h5')


