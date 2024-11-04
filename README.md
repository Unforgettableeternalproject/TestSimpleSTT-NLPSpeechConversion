# TestSimpleSTT-NLPSpeechConversion

## Overview

This project consists of three main components: DistilBERT model training, real-time speech-to-text (STT), and natural language processing (NLP) using the trained model. The goal is to classify spoken commands, chat, and non-sense in real-time.

## Features

- **DistilBERT Model Training**: Train a DistilBERT model to classify text into three categories: command, chat, and non-sense.
- **Real-Time Speech-to-Text (STT)**: Convert spoken language into text using a microphone in real-time.
- **NLP Processing**: Classify the transcribed text using the trained DistilBERT model.

## Project Structure



```graphql
│ SimpleNLP-STT-DistilBERT/ 
├── train/ 
│   └── dbert_train.py
│
├── message_handler/
│   ├── simpleNLP.py
│   └── stt.py
│
├── Entry.py
└── README.md
```


## Components

### DistilBERT Model Training

The `dbert_train.py` script is responsible for training the DistilBERT model. It includes data loading, preprocessing, tokenization, and model training. The trained model is saved for later use in NLP processing.

### Real-Time Speech-to-Text (STT)

The `stt.py` script handles real-time speech-to-text conversion using the `speech_recognition` library. It captures audio from a microphone, transcribes it to text, and places the text in a queue for NLP processing.

### NLP Processing

The `simpleNLP.py` script uses the trained DistilBERT model to classify the transcribed text into one of three categories: command, chat, or non-sense. It integrates with the STT component to process real-time transcriptions.

---

## Disclaimer

This project is **not** intended for public use. 

The model within this project is not public, and the server IP and port are fake. 

For most part, it is for demonstration purposes only.
