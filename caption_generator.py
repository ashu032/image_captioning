import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Dropout, LSTM, Embedding
from keras.models import Model
from keras.utils import plot_model
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from keras.utils import to_categorical

def datagen(captions, features, tok, max_length, total_words):
    while True:
        for key, caplist in captions.items():
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tok, max_length, caplist, feature, total_words)
            yield ([input_image, input_sequence], output_word)

def create_sequences(tok, max_length, caplist, feature, total_words):
    feat, input_seq, output_seq = [], [], []
    for caps in caplist:
        sequence = tok.texts_to_sequences([caps])[0]
        for i in range(1, len(sequence)):
            iseq = pad_sequences([sequence[:i]], maxlen=max_length)[0]
            oseq = to_categorical([sequence[i]], num_classes=total_words)[0]
            feat.append(feature)
            input_seq.append(iseq)
            output_seq.append(oseq)
    X1 = np.array(feat)
    X2 = np.array(input_seq)
    y = np.array(output_seq)
    return X1, X2, y

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    try:
        if image.shape[2] == 4: 
            image = image[..., :3]
    except:
        print(f"{image.shape}")
    try:
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image, verbose=0)
    except:
        return None
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        
        if word is None:
            break
        in_text += ' ' + word
        
        if word == 'end':
            break
    return in_text
