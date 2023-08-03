# Automated Image Captioning Using Multimodal Contextual Cues

This project implements an image captioning model with attention mechanism using the Xception model as the image encoder. The model generates captions for images by attending to different parts of the image while generating each word in the caption.

## Requirements

- Python 3.6 or above
- TensorFlow 2.x
- Keras
- NumPy
- pandas
- scikit-image
- Matplotlib
- Seaborn
- Natural Language Toolkit (NLTK)

## Usage

1. Install the required dependencies by running:

```bash
  pip install -r requirements.txt
  ```

2. Place the `captions_train2017.json` and `captions_val2017.json` files into the `annotations/` directory. The training and validation images should be placed in the `train2017/` and `val2017/` directories, respectively.

3. Modify the `BASE_PATH` variable in `main.py` to point to the base directory containing the `annotations/`, `train2017/`, and `val2017/` directories.

4. To train the model, run the following command:

```bash
    python main.py
    ```




