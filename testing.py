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

# Загрузка токенизатора 
with open('output/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 5. Тестирование модели на небольшом предложении
test_sentence = ['столов', 'низк', 'потолк', 'глубок', 'земл',]




# Предсказание следующего слова для каждой модели
for size in params.embedding_sizes:
    # Загрузка модели
    model = load_model(f"output/model_trained{size}.keras")
    # Загрузка word2vec
    word2vec_model = Word2Vec.load(f"output/word2vec{size}.model")
    # Заменяем слова на векторы
    context_vectors = [word2vec_model.wv[word] for word in test_sentence]
    input_data = np.array([context_vectors])  
    # Предсказание
    predictions = model.predict(input_data)
    predicted_idx = np.argmax(predictions[0])
    predicted_word = tokenizer.index_word.get(predicted_idx, '<UNKNOWN>')
    print(f"\nWith embedding size {size}, predicted next word: '{predicted_word}'")

