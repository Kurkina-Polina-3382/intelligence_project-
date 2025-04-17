import numpy as np                   # продвинутая математическая библиотека
import matplotlib.pyplot as plt      # библиотека для рисования графиков
import random       
import params as params     

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import pickle
from preprosessing_text import preprocess_text

# Загрузка словаря
with open('output/vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = vocab_data['idx_to_word']

# 5. Тестирование модели на небольшом предложении
test_sentence = "В столовой с низким потолком, глубоко под землей,"
test_tokens = preprocess_text(test_sentence)
test_indices = [word_to_idx[word] for word in test_tokens]


# Предсказание следующего слова для каждой модели
for size in params.embedding_sizes:
    # Загрузка модели
    model = load_model(f"output/model_trained{size}.keras")
    # Загрузка word2vec
    embeddings = np.load(f'output/word2vec_embeddings_{size}.npy')
    # Заменяем слова на векторы
    input_data = np.array([[embeddings[word_to_idx[word]] for word in test_tokens[0]]])

    # Предсказание
    predictions = model.predict(input_data)
    predicted_idx = np.argmax(predictions[0])
    predicted_word = idx_to_word[predicted_idx]
    print(f"\nWith embedding size {size}, predicted next word: '{predicted_word}'")

