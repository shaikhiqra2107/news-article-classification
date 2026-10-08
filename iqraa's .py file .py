
# Classifying News Articles Project
# Author: ChatGPT (Generated)
# ---------------------------------------------
# Goal: Classify news headlines into 5-7 categories using Logistic Regression & LSTM
# ---------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
DATA_PATH = "news_dataset.csv"  # path to your dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found. Please add 'news_dataset.csv' to this directory.")

df = pd.read_csv(DATA_PATH)
print("Dataset Loaded:", df.shape)

# ---------------------------------------------
# Preprocessing
# ---------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

label_col = [c for c in df.columns if "category" in c.lower() or "section" in c.lower()][0]
text_col = [c for c in df.columns if "headline" in c.lower() or "title" in c.lower() or "text" in c.lower()][0]

TARGET_CATEGORIES = ['POLITICS', 'SPORTS', 'TECH', 'BUSINESS', 'ENTERTAINMENT']

df['label_upper'] = df[label_col].astype(str).str.upper()
df = df[df['label_upper'].isin(TARGET_CATEGORIES)].copy()
df['clean_text'] = df[text_col].astype(str).apply(clean_text)
df = df[df['clean_text'].str.strip() != '']

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label_upper'])

print("Filtered Dataset:", df.shape)
print("Classes:", list(le.classes_))

# ---------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------
X = df['clean_text']
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

print("\nTraining TF-IDF + Logistic Regression Model...")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ---------------------------------------------
# LSTM Model
# ---------------------------------------------
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 100
GLOVE_PATH = "glove.6B.100d.txt"

print("\nPreparing data for LSTM...")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQUENCE_LENGTH)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQUENCE_LENGTH)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

word_index = tokenizer.word_index
print("Total tokens:", len(word_index))

embeddings_index = {}
if not os.path.exists(GLOVE_PATH):
    print("GloVe not found, continuing with random embeddings.")
else:
    print("Loading GloVe vectors...")
    with open(GLOVE_PATH, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Loaded", len(embeddings_index), "word vectors.")

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential([
    Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("\nTraining LSTM Model...")
history = model.fit(
    X_train_seq, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

loss, acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
print(f"\nLSTM Test Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("LSTM Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nSummary:")
print("- Logistic Regression Accuracy: {:.2f}%".format(pipe.score(X_test, y_test)*100))
print("- LSTM Accuracy: {:.2f}%".format(acc*100))
