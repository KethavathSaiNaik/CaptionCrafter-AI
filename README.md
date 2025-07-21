
# 📸 CaptionCrafter-AI

**CaptionCrafter-AI** is an intelligent image captioning system powered by deep learning. It generates natural language captions for input images using a fusion of **EfficientNetB0** image features and **LSTM-based sequence modeling**.
##  Project Highlights

- 📷 Image encoder using **EfficientNetB0**
- 📝 Text input embedded using a custom tokenizer
- 🔗 Early fusion of image and text followed by **LSTM**
- 🔁 Residual connection (image features added back post-LSTM)
- 🔮 Trained on standard captioning datasets (Flickr30k)
##  Model Architecture

The model takes two inputs:

- **Image feature vector**: Extracted using EfficientNetB0 and reduced to 256 dimensions
- **Text sequence input**: Tokenized caption padded to fixed length and embedded

These two are concatenated into a single sequence and passed through an LSTM. A skip connection from image features is added back after the LSTM to enrich context before final prediction.

## ⚙️ How It Works

1. **Feature Extraction**:  
   EfficientNetB0 extracts a 1280D feature vector from input images.

2. **Caption Tokenization**:  
   Captions are preprocessed using a custom tokenizer and padded to fixed length.

3. **Early Fusion**:  
   Image and caption embeddings are concatenated and passed into an LSTM.

4. **Prediction**:  
   The model predicts the next word token using a softmax layer over the vocabulary.

5. **Inference Loop**:  
   At test time, the model auto-regressively generates the caption word-by-word.



### Installation

Clone the Repository

```bash
  git clone https://github.com/KethavathSaiNaik/CaptionCrafter-AI.git
  cd captioncrafter-ai
```
Create and Activate a Virtual Environment
```bash
  python -m venv venv
  venv\Scripts\activate
```
 Install Dependencies


```bash
  pip install -r requirements.txt
```
Launch Jupyter Notebook
```bash
  jupyter notebook
```
Then open image-caption-generator.ipynb and follow the steps to:

Preprocess data

Train or load the model

Generate captions for new images

    
