import numpy as np
from PIL import Image
import cv2
from keras.applications.xception import Xception, preprocess_input
from tqdm.auto import tqdm

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct.")
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