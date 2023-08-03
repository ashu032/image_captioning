import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from caption_generator import extract_features, generate_desc

def evaluate_model(val_dataset, tokenizer, encoder, max_length, model):
    bleu_scores = {}
    met_scores = {}

    for img_path, captions in val_dataset.items():
        photo = extract_features(img_path, encoder)
        if photo is None:
            continue
        generated_caption = generate_desc(model, tokenizer, photo, max_length)
        generated_caption = generated_caption.split()[1:-1]  
        references = [caption.split()[1:-1] for caption in captions]  
        bleu_score = sentence_bleu(references, generated_caption)
        bleu_scores[img_path] = bleu_score

        met_score = meteor_score(" ".join(references[0]), " ".join(generated_caption))
        met_scores[img_path] = met_score

    return bleu_scores, met_scores
