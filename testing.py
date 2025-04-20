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


def calculate_perplexity(model, embeddings, word_to_idx, test_sequence, L=params.L):
    log_probs = []
    for i in range(L, len(test_sequence)):
        context = test_sequence[i - L:i]
        target_word = test_sequence[i]
        context_vectors = [embeddings[word_to_idx.get(word, 1)] for word in context]
        context_vectors = np.array(context_vectors).reshape(1, L, -1)
        predictions = model.predict(context_vectors, verbose=0)[0]
        target_index = word_to_idx.get(target_word, 1)
        target_prob = predictions[target_index]
        if target_prob > 0:
            log_probs.append(np.log(target_prob))
        else:
            log_probs.append(-np.inf)
    perplexity = np.exp(-np.mean(log_probs))
    return perplexity


test_sentence = "лицо ее вдруг представило глубокое и искреннее выражение"
test_tokens = preprocess_text(test_sentence)

text_perplexity = "Быть энтузиасткой сделалось ее общественным положением. "
text_perp_tokens = preprocess_text(text_perplexity)


for size in params.embedding_sizes:
    # Загрузка модели
    model = tf.keras.models.load_model(f"output/model_trained{size}.keras")
    # Загрузка word2vec
    embeddings = np.load(f'output/word2vec_embeddings_{size}.npy')
    perplexity = calculate_perplexity(model, embeddings, word_to_idx, text_perp_tokens[0], L=params.L)
    print(f"Модель с размером эмбендинга {size}: Перплексия = {perplexity:.6f}")
    # Заменяем слова на векторы
    input_data = np.array([[embeddings[word_to_idx[word]] for word in test_tokens[0]]])

    # Предсказание
    predictions = model.predict(input_data)
    predicted_idx = np.argmax(predictions[0])
    predicted_word = idx_to_word[predicted_idx]
    print(f"\nWith embedding size {size}, predicted next word: '{predicted_word}'")