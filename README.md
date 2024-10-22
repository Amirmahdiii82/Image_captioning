# Image Captioning


---

## Project Overview

Automatically generating descriptive captions for images is a challenging task with applications in accessibility, image retrieval, and content generation. This project focuses on building a deep learning model using the **Flickr8k** dataset to generate accurate captions for images by combining **Convolutional Neural Networks (CNNs)** for image feature extraction and **Recurrent Neural Networks (RNNs)** for language generation.

---

## Problem Statement

Generating meaningful image captions requires effectively handling various challenges such as:

1. **Handling Diversity**: Dealing with diverse image types, scenes, and objects.
2. **Model Complexity**: Balancing model performance and computational efficiency.
3. **Limited Dataset Size**: Overcoming limitations posed by the relatively small size of the **Flickr8k** dataset.
4. **Evaluation Metrics**: Choosing appropriate evaluation metrics for the generated captions.
5. **Cross-Modal Learning**: Aligning image and textual data effectively.


---

## Model Architecture

The model consists of two major components:

1. **Encoder (ResNet-50)**: Extracts visual features from images.
2. **Decoder (LSTM)**: Generates sequential captions based on image features.

Both components work together in a unified model to generate meaningful captions for given images.

---

## Dataset

The **Flickr8k** dataset, containing 8,000 images and 40,000 human-written captions, is used for training and evaluation. The dataset is split into training (6,000 images), validation (1,000 images), and test (1,000 images) sets.

---

## Training

The model was trained using a combination of CNN and LSTM with the following hyperparameters:

- **Embedding Dimension**: 300
- **Hidden Dimension**: 500
- **LSTM Layers**: 2
- **Batch Size**: 128
- **Learning Rate**: 1.25
- **Optimizer**: SGD

---

## Results

The model's performance was evaluated using BLEU scores:

- **BLEU-1**: 0.6108
- **BLEU-2**: 0.3907
- **BLEU-3**: 0.2519
- **BLEU-4**: 0.1636

The model performs well on unigram-based evaluation (BLEU-1) but shows room for improvement in generating longer, coherent phrases (BLEU-4).

---

## Conclusion

While the model demonstrates good performance in generating relevant individual words, future improvements could involve enhancing the **LSTM layers** or fine-tuning the **ResNet** model to generate more natural and contextually accurate captions.
