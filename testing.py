import numpy as np                   # продвинутая математическая библиотека
import matplotlib.pyplot as plt      # библиотека для рисования графиков
import random            

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import pickle

embedding_sizes = [100, 500, 1000]
# Загрузка токенизатора
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 5. Тестирование модели на небольшом предложении
test_sentence = "Описывает русское общество в"
test_sequence = tokenizer.texts_to_sequences([test_sentence])[0]
test_sequence = pad_sequences([test_sequence], maxlen=L, padding='pre')

# Предсказание следующего слова для каждой модели
for size in embedding_sizes:
    # Загрузка модели
    loaded_model = load_model(f"model_trained{size}.keras")  # пример для size=32

    # Предсказание
    predictions = loaded_model.predict(test_sequence)

    decoded_text = tokenizer.sequences_to_texts([predictions[0]])[0]  
    clean_text = ' '.join([word for word in decoded_text.split() if word != '0'])  
    print(f"\nWith embedding size {size}, predicted next word: '{clean_text}'")
    
    print()