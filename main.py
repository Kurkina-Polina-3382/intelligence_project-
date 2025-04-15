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

text = """
Война и мир - роман Льва Толстого, одно из величайших произведений мировой литературы. 
Описывает русское общество в эпоху войн против Наполеона. 
Главные герои: Пьер Безухов, Андрей Болконский, Наташа Ростова.
"""

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

# Сохранение
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# 2. Построение эмбедингов Word2Vec с разными размерностями
sentences = [text.split()]  # преобразуем текст в список предложений (токенов)

embedding_sizes = [100, 500, 1000]
word2vec_models = {}

for size in embedding_sizes:
    model = Word2Vec(sentences, vector_size=size, window=5, min_count=1, workers=4)
    word2vec_models[size] = model

# Функция для преобразования слова в вектор с помощью Word2Vec
def word_to_vec(word, embedding_size):
    return word2vec_models[embedding_size].wv[word]

# 3. Подготовка данных для нейросети
L = 5  # количество предыдущих слов для предсказания следующего
X = []
y = []

for i in range(L, len(sequences)):
    X.append(sequences[i-L:i])
    y.append(sequences[i])

X = np.array(X)
y = np.array(y)
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# 4. Создание и обучение нейросети
def create_model(embedding_size):
    model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=L),
        Flatten(),
        Dense(500, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Обучаем модели с разными эмбедингами
history_dict = {}

for size in embedding_sizes:
    print(f"\nTraining model with embedding size {size}")
    model = create_model(size)
    history = model.fit(X, y, epochs=50, batch_size=2, verbose=1)
    history_dict[size] = history.history['loss']

    # Сохраняем модель в файл
    model.save(f"model_trained{size}.keras")  
    print(f"Model with embedding size {size} saved!")

# Визуализация потерь при обучении
plt.figure(figsize=(10, 6))
for size, losses in history_dict.items():
    plt.plot(losses, label=f'Embedding size {size}')

plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
#plt.show() # просто не хочу чтоб он рисовал достал меня
plt.savefig("losses.jpg")

