# ========================================
# ========== keras version code ==========
# ========================================
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization, LSTM, ConvLSTM2D, Reshape
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D, MaxPooling3D, GlobalMaxPooling3D
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

# model definition
main_inputs = Input(shape = (None, 270, 480, 1))

conv1 = Conv3D(filters = 256, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu')(main_inputs)
pool1 = MaxPooling3D()(conv1)
bn1   = BatchNormalization()(pool1)

conv2 = Conv3D(filters = 200, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu')(bn1)
pool2 = MaxPooling3D()(conv2)
bn2   = BatchNormalization()(pool2)

conv3 = Conv3D(filters = 144, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu')(bn2)
pool3 = MaxPooling3D()(conv3)
bn3   = BatchNormalization()(pool3)

GM = GlobalMaxPooling3D()(bn3)

reshape = Reshape((12, 12), input_shape = (None, 144))(GM)

lstm1 = LSTM(units = 64, return_sequences = True)(reshape)
flat = Flatten()(lstm1)
hidden = Dense(128, activation = 'relu')(flat)
output = Dense(1, activation = 'linear')(hidden)

print('1.model definition')
model = Model(main_inputs, output)
print(model.summary())

# model compile
model.compile(loss = 'sparse_categorical_crossentropy', # <- because we are dealing with integer labels!
              optimizer = 'adam',
              metrics = ['accuracy'])
print('2.model compile')

# ========== callback functions ==========
import keras
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 7),
    keras.callbacks.ModelCheckpoint(filepath = "Keras_best_model.h5", monitor = 'val_loss', save_best_only = True),
    keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5)]


csv_path = '/DATA/trainset-for_user.csv'
df = pd.read_csv(csv_path, encoding = 'utf-8', names = ['folder_name_list', 'img_path_list', 'label'])
df['new_label'] = df['label'].apply(convert_label)
print('3.dataframe:')
print(df)


folder_name_list = list(set(df['folder_name_list'])) # duplicate removed

data = {}
for each_folder_name in folder_name_list:
    data[each_folder_name] = [ [], [] ]
print('4.len(data):', len(data))

print('5. allocation data and traininig part start')
total_image_sequence = []
total_label_sequence = []
for index, each_folder_name in tqdm(enumerate(folder_name_list[:200])):

    img_path_list   = np.array(df[df['folder_name_list'] == each_folder_name]['img_path_list'])
    label_list      = np.array(df[df['folder_name_list'] == each_folder_name]['new_label'])

    image_seq = []
    label_seq = []
    for each_img_path, each_label in tqdm(zip(img_path_list, label_list)):

        path = os.path.join('/DATA/' + each_folder_name, each_img_path)
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
    label_seq = np.array(label_seq) # 1 dimension
    print('image_seq shape:', image_seq.shape)
    print('label_seq shape:', label_seq.shape)

    hist = model.fit(image_seq, label_seq,
                     epochs = 20,
                     callbacks = callbacks_list)

    total_image_sequence.append(image_seq)
    total_label_sequence.append(label_seq)

total_image_sequence = np.array(total_image_sequence)
total_label_sequence = np.array(total_label_sequence)

print('total_image_sequence :', total_image_sequence.shape)
print('total_label_sequence :', total_label_sequence.shape)


np.save('total_image_sequence_200', total_image_sequence)
np.save('total_label_sequence_200', total_label_sequence)
print('saved')


#print('{}-th training start'.format(index + 1))
#hist = model.fit(total_image_sequence, total_label_sequence,
#                 batch_size = 32,
#                 epochs = 20,
#                 callbacks = callbacks_list,
#                 validation_split = 0.2)
#print('{}-th training done'.format(index + 1))

print('6. done!')
                        