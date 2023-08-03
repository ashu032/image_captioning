import json
import pandas as pd
import re
import collections

def load_data(base_path):
    with open(f'{base_path}/annotations/captions_train2017.json', 'r') as f:
        data = json.load(f)
        data = data['annotations']

    with open(f'{base_path}/annotations/captions_val2017.json', 'r') as f:
        val_data = json.load(f)
        val_data = val_data['annotations']

    img_cap_pairs = []
    val_img_cap_pairs = []

    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_cap_pairs.append([img_name, sample['caption']])

    for sample in val_data:
        img_name = '%012d.jpg' % sample['image_id']
        val_img_cap_pairs.append([img_name, sample['caption']])

    df = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
    val_df = pd.DataFrame(val_img_cap_pairs, columns=['image', 'caption'])

    df['image'] = df['image'].apply(lambda x: f'{base_path}/train2017/{x}')
    val_df['image'] = val_df['image'].apply(lambda x: f'{base_path}/val2017/{x}')

    train_df = df.sample(70000)
    train_df = train_df.reset_index(drop=True)

    return train_df, val_df

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = '[start] ' + text + ' [end]'
    return text

def create_dataset(dataframe):
    dataset = collections.defaultdict(list)
    for img, cap in zip(dataframe['image'], dataframe['caption']):
        dataset[img].append(cap)
    return dataset

def create_validation_dataset(dataframe):
    val_dataset = collections.defaultdict(list)
    for img, cap in zip(dataframe['image'], dataframe['caption']):
        val_dataset[img].append(cap)
    return val_dataset
