# ========================================
# ========== keras version code ==========
# ========================================
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate

import csv
import os
import numpy as np
import pickle
import pandas as pd
from torch.utils import data
from torchvision import datasets, transforms
import random
from multiprocessing import cpu_count
import torch
from PIL import Image
from tqdm import tqdm

def convert_label(label):
    if label == 'Wake' : label = 0
    elif label == 'N1' : label = 1
    elif label == 'N2' : label = 2
    elif label == 'N3' : label = 3
    elif label == 'REM' : label = 4
    else: label = None
    return label

def transform_image(image):
    custom_transformer = transforms.Compose([transforms.ToTensor(), ])
    image_transform = custom_transformer(image)
    return image_transform


def build_model():
    # input layer
    main_inputs = Input(shape = (270, 480, 1)) # channel dimension last

    conv1 = Conv2D(64, kernel_size = 5, activation = 'relu')(main_inputs)
    pool1 = MaxPooling2D(pool_size = 2)(conv1)
    bn1   = BatchNormalization()(pool1)

    conv2 = Conv2D(64, kernel_size = 5, activation = 'relu')(bn1)
    pool2 = MaxPooling2D(pool_size = 2)(conv2)
    bn2   = BatchNormalization()(pool2)

    GMpool = GlobalMaxPooling2D()(bn2)

    hidden1 = Dense(64, activation = 'relu')(GMpool)
    hidden2 = Dense(32,  activation = 'relu')(hidden1)

    main_outputs = Dense(5, activation = 'softmax')(hidden2)

    # definition of model with main_inputs and main_outputs
    model = Model(inputs = main_inputs,
                  outputs = main_outputs)
    # summarize layers
    print(model.summary())
    return model


csv_path = '/DATA/trainset-for_user.csv'
df = pd.read_csv(csv_path, encoding = 'utf-8', names = ['folder_name_list', 'img_path_list', 'label'])
df['new_label'] = df['label'].apply(convert_label)
print('1. dataframe:')
print(df)



folder_name_list = list(set(df['folder_name_list'])) # duplicate removed

data = {}
for each_folder_name in folder_name_list:
    data[each_folder_name] = [ [], [] ]

print('2. len(data):', len(data))




# definition of model
model = build_model()
print('3. model builded')

# model compile
model.compile(loss = 'sparse_categorical_crossentropy', # <- because we are dealing with integer labels!
              optimizer = 'adam',
              metrics = ['accuracy'])
print('4. compile')

# ========== callback functions ==========
import keras
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 7),
    keras.callbacks.ModelCheckpoint(filepath = "Keras_best_model.h5", monitor = 'val_loss', save_best_only = True),
    keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5)]



print('5. allocation data and traininig part start')
for index, each_folder_name in tqdm(enumerate(folder_name_list)):

    img_path_list   = np.array(df[df['folder_name_list'] == each_folder_name]['img_path_list'])
    label_list      = np.array(df[df['folder_name_list'] == each_folder_name]['new_label'])

    image_seq = []
    label_seq = []
    for each_img_path, each_label in tqdm(zip(img_path_list, label_list)):

        path = os.path.join('/DATA/' + each_folder_name, each_img_path)
        print(path)
        try:
            img = Image.open(path)
            img = transform_image(img) # to tensor
            img = np.array(img) # to numpy
            img = img.reshape(270, 480, 1)
            image_seq.append(img)
            label_seq.append(each_label)
        except Exception:
            print('No such file or directory')

    image_seq = np.array(image_seq) # 4 dimenison
    label_seq = np.array(label_seq) # 2 dimension
    print('image_seq shape:', image_seq.shape)
    print('label_seq shape:', label_seq.shape)

    print('{}-th training start'.format(index + 1))
    hist = model.fit(image_seq, label_seq,
                     batch_size = 32,
                     epochs = 20,
                     callbacks = callbacks_list,
                     validation_split = 0.2)
    print('{}-th training done'.format(index + 1))


print('6. done!')

