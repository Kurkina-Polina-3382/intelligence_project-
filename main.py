import numpy as np   
import re               
import os
import matplotlib.pyplot as plt  
from collections import Counter   
from multiprocessing import Pool  
   
import pickle       
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer


from preprosessing_text import preprocess_text
import params as params


def load_learning_data():
    print("Подготовка текста \n\n")
    learning_files = os.listdir("learning_data")
    data = []  
    for f in learning_files:
        with open("learning_data/" + f, 'r', encoding='utf-8') as file:
            text = file.read()
            data.extend(preprocess_text(text))

    return data

text = load_learning_data()

# словарь 

def build_vocab(texts):
    word_to_id = {}
    id_to_word = {}
    for sentance in text:
        for i, token in enumerate(set(sentance)):
                word_to_id[token] = i
                id_to_word[i] = token
    
    return word_to_id, id_to_word

word_to_idx, idx_to_word = build_vocab(text)
vocab_size = len(word_to_idx)
# сохраняем словари
with open('output/vocab.pkl', 'wb') as f:
    pickle.dump({
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,

    }, f, protocol=pickle.HIGHEST_PROTOCOL)


# эмбединги
# генерация данных для обучения Word2Vec
def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res
def generate_training_data(text, word_to_id, L):
    X = []
    y = []
    for tokens in text:
        n_tokens = len(tokens)

        for i in range(n_tokens):
            idx = concat(
                range(max(0, i - L), i),
                range(i, min(n_tokens, i + L + 1))
            )
            for j in idx:
                if i == j:
                    continue
                X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
                y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(X), np.asarray(y)


# Реализация Word2Vec
def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model
def forward(model, X, return_cache=True):
    cache = {}

    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    cache["z"] = np.clip(cache["z"], 1e-10, 1.0)

    if not return_cache:
        return cache["z"]
    return cache

def softmax(X):
    
    res = []
    for x in X:
        e_x = np.exp(x - np.max(x))
        #exp = np.exp(x)
        # if(exp.sum() <= 0 ):
        #     print(f"Сумма exp равна нулю для x = {x}", flush=True)
        #     print(X,  flush=True)
        #     exit()
        #assert(exp.sum() > 0)
        res.append(e_x / e_x.sum())
    return res

def backward(model, X, y, alpha):
    cache = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

def cross_entropy(z, y):
#    for i in range(len(z)):
#        for b in range(len(z[i])):
#             if z[i][b] == 0:
#                 print("ОШИБКА: ", i, b)
#                 print("логарифм ",np.log(z[i][b]))
#                 assert(z[i][b] != 0)
   #assert(len(np.nonzero(z==0)) != False)
   return - np.sum(np.log(z) * y)

# обучение Word2Vec
trained_models = {}
for size in params.embedding_sizes:
    print(f"обучение word2vec с размером эмбэддинга {size}\n")
    X, y = generate_training_data(text, word_to_idx, params.L)
    model = init_network(len(word_to_idx), size)
    history = [backward(model, X, y, params.learning_rate) for _ in range(params.epochs_word2vec)]
    # сохраняем эмбэддинги
    # np.save(f'output/word2vec_embeddings_{size}.npy', W_input)
    trained_models[size] = model


# подготовка данных для нейросети
def prepare_data(texts, embeddings, tokenizer, L=params.L):
    X, y = [], []
    for sentence in texts:
        for i in range(len(sentence) - L):
            context = sentence[i:i+L]
            target = sentence[i+L]

            # слова в эмбеддинги
            context_vectors = []
            for word in context:
                idx = word_to_idx.get(word, 1)
                context_vectors.append(embeddings.get(idx, 1))
            #context_vectors = [embeddings[word_to_idx.get(word, 1)] for word in context]
            X.append(context_vectors)
            y.append(word_to_idx.get(target, 1))

    # целевые слова в категориальные метки
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
    
    return np.array(X), np.array(y)


 #создание нейросети
def create_model(embedding_size):
    model = Sequential([
        Input(shape=(params.L, embedding_size)),
        Flatten(),
        Dense(1500, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

history_dict = {}
# обучение модели с разными эмбедингами
for size in params.embedding_sizes:
    
    print(f"Запущено обучение модели с эмбэддингом {size}")
    X, y = prepare_data(text, trained_models[size], None, L=params.L)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = create_model(size)
    history = model.fit(X_train, y_train, epochs=params.epochs, batch_size=params.batch_size)
    history_dict[size] = history.history['loss']

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Размер Эмбэддинга {size}: Точность на тестирующей выборке = {accuracy:.4f},  Потери = {loss:.4f}")
    model.save(f"output/model_trained{size}.keras")
    
    # предсказываем слово

    test_sentence = "В столовой с низким потолком, глубоко под землей,"
    test_tokens = preprocess_text(test_sentence)
    embeddings = trained_models[size]
    input_vector = np.array([[embeddings[word_to_idx[word]] for word in test_tokens[0]]])
    # предсказание
    output = model.predict(input_vector)
    predicted_word = idx_to_word[np.argmax(output)]
    print(f"Предсказанное слово моделью {size} : {predicted_word}")

# визуализация потерь при обучении
plt.figure(figsize=(10, 6))
for size, losses in history_dict.items():
    plt.plot(losses, label=f'Embedding size {size}')

plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show() 
plt.savefig("output/losses.jpg")


