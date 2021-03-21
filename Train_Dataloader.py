
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
for index, each_folder_name in tqdm(enumerate(folder_name_list[:600])):

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

    total_image_sequence.append(image_seq)
    total_label_sequence.append(label_seq)

total_image_sequence = np.array(total_image_sequence)
total_label_sequence = np.array(total_label_sequence)

print('total_image_sequence :', total_image_sequence.shape)
print('total_label_sequence :', total_label_sequence.shape)

np.save('train_image_sequence_600', total_image_sequence)
np.save('train_label_sequence_600', total_label_sequence)
print('saved')


