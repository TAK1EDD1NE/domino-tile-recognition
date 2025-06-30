# from google.colab import drive
# drive.mount('/content/drive')
# !pip install keras.preprocessing
import numpy as np
import pandas as pd
import os
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
keras.__version__
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Check if GPU is available
# print("GPU Available: ", tf.config.list_physical_devices('GPU'))
# print("Built with CUDA: ", tf.test.is_built_with_cuda())

# # Configure GPU memory growth to avoid OOM errors
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
#     except RuntimeError as e:
#         print(e)

# # Verify GPU is being used
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # Set mixed precision for better performance on T4
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print('Mixed precision enabled:', policy.name)


# DATA_DIRECTORY = os.path.abspath("/content/drive/MyDrive/data")
DATA_DIRECTORY = os.path.abspath("C:\\Users\\pc\\Desktop\\domino recognition\\archive\\data")

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_SEED=42

np.random.seed(RANDOM_SEED)
def categorized_from_directory(path):
    """Returns a Pandas dataframe with the `category` and `path` of each image."""
    rows = []
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        for image in os.listdir(category_path):
            image_path = os.path.join(category_path, image)
            rows.append({'category': category, 'path': image_path})
    return pd.DataFrame(rows)

all_classes = [f"{i}x{j}" for i in range(7) for j in range(0, i + 1)]
print(all_classes)

full_data = categorized_from_directory(DATA_DIRECTORY)

# Put aside a test set for final evaluation
train_data, test_data = train_test_split(
    full_data,
    test_size=TEST_SPLIT,
    stratify=full_data['category'])

# Further decompose training data for training and validation
train_data, validation_data = train_test_split(
    train_data,
    test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
    stratify=train_data['category'])
print("train_data :",train_data, "__"*10,"val_data :", validation_data, "__"*10,"test_data :", test_data )
num_categories = len(full_data['category'].unique())

assert num_categories == len(all_classes)

num_categories
train_data.head()


len(train_data), len(validation_data), len(test_data)
print(train_data.groupby('category').count())
print(validation_data.groupby('category').count())
print(test_data.groupby('category').count())
BATCH_SIZE = 20
IMAGE_SIZE = (100, 100)

def flow_from_datagenerator(datagen, data, batch_size=BATCH_SIZE, shuffle=True):
    """Returns a generator from an ImageDataGenerator and a dataframe."""
    return datagen.flow_from_dataframe(
        dataframe=data,
        x_col="path",
        y_col="category", class_mode='categorical',
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        shuffle=shuffle,
        classes=all_classes)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=360,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    brightness_range=(-0.1, 0.1),
    #shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False)

train_generator = flow_from_datagenerator(train_datagen, train_data)

train_steps = train_generator.n // train_generator.batch_size
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = flow_from_datagenerator(validation_datagen, validation_data)

validation_steps = validation_generator.n // validation_generator.batch_size
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = flow_from_datagenerator(test_datagen, test_data)

test_steps = test_generator.n // test_generator.batch_size

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_categories, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
model.summary()
# EPOCHS = 1200
EPOCHS = 1000

checkpoint = callbacks.ModelCheckpoint(
    "best_model_1000_epochs.keras", 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    mode='max')

history = model.fit(
    train_generator, 
    steps_per_epoch=train_steps, 
    epochs=EPOCHS, 
    validation_data=validation_generator, 
    validation_steps=validation_steps,
    callbacks=[checkpoint]
    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()