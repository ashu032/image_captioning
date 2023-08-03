import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import cv2
from skimage.io import io
import tqdm
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from data_loader import load_data, preprocess_text, create_dataset, create_validation_dataset
from caption_generator import datagen
from evaluate import evaluate_model
from model import define_model

BASE_PATH = 'path/to/directory'

train_df, val_df = load_data(BASE_PATH)

train_df['caption'] = train_df['caption'].apply(preprocess_text)
val_df['caption'] = val_df['caption'].apply(preprocess_text)

dataset = create_dataset(train_df)
val_dataset = create_validation_dataset(val_df)

encoder = Xception(include_top=False, pooling='avg')

image_features = {}
for img in tqdm(dataset.keys()):
    image = io.imread(img)
    if image.ndim != 3:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image,(299,299))
    image = np.expand_dims(image, axis=0)

    image = image/127.5
    image = image - 1.0

    feature = encoder.predict(image, verbose=0)
    image_features[img] = feature

print(f"{len(dataset)} images processed")

flatten_list = list(chain.from_iterable(dataset.values()))
tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(flatten_list)
total_words = len(tokenizer.word_index) + 1

print("Vocabulary length:", total_words)

max_length = max(len(d.split()) for d in flatten_list)
print("Max caption length:", max_length)

steps = len(dataset)
generator = datagen(dataset, image_features, tokenizer, max_length, total_words)
model = define_model(total_words, max_length)
history = model.fit(generator, epochs=50, steps_per_epoch=steps)

bleu_score, met_score = evaluate_model(val_dataset, tokenizer, encoder, max_length, model)
all_scores = [bleu_score.values(), met_score.values()]

ax = sns.violinplot(all_scores)
ax.set_xticklabels(['BLEU score', 'METEOR score'])
plt.title('Scores Distribution')
plt.show()
