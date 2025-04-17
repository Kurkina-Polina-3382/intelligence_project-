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



# def preprocess_text(text):
    
#     data = []  

#     stemmer = SnowballStemmer("russian")
#     stop_words = set(stopwords.words('russian'))

#     sentences = sent_tokenize(text, language='russian')  # преобразуем текст в список предложений (токенов)
    
#     for sent in sentences:
#         # Приведение к нижнему регистру
#         sent = sent.lower()
            
#         # Удаление спецсимволов и цифр
#         sent = re.sub(r'[^а-яёa-z\s]', '', sent) 
        
        
#         # Токенизация
#         tokens = word_tokenize(sent, language='russian')


#         # Удаление стоп-слов
#         tokens = [word for word in tokens if word not in stop_words]
        
#         # Стемминг
#         # но вообще можно использовать лемматизацию. она должна давать лучше результат (бежал - бегать) morph = MorphAnalyzer()
#         tokens = [stemmer.stem(word) for word in tokens]
        
#         data.append(tokens)  # Добавляем список токенов (предложение)
#         #может и не делать разбиение на предложение
                
#     # Фильтрация от пустых предложений
#     data = [sent for sent in data if sent]
#     return data

def load_learning_data():
    print("preprocessing text")
    learning_files = os.listdir("learning_data")
    data = []  
    for f in learning_files:
        with open("learning_data/" + f, 'r', encoding='utf-8') as file:
            text = file.read()
            data.extend(preprocess_text(text))

    return data

text = load_learning_data()

# 2. Создание словаря


def build_vocab(texts):
    # 1. Подсчет и автоматическая сортировка слов по частоте
    vocab = [word for word, _ in Counter(word for sentence in texts for word in sentence).most_common(params.vocab_size)]
    
    # 2. Добавляем служебные токены
    vocab = ['<PAD>', '<OOV>'] + vocab
    
    # 3. Создаем словари
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    return word_to_idx, idx_to_word, len(vocab)

word_to_idx, idx_to_word, vocab_size = build_vocab(text)

# сохраняем словари
with open('output/vocab.pkl', 'wb') as f:
    pickle.dump({
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size
    }, f, protocol=pickle.HIGHEST_PROTOCOL)


# эмбединги
# 3. Генерация данных для обучения Word2Vec
def generate_training_data(texts, word_to_idx, window_size=5):
    data = []
    for sentence in texts:
        for i, target_word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for context_word in sentence[start:i] + sentence[i+1:end]:
                data.append((word_to_idx[target_word], word_to_idx[context_word]))
    return data

training_data = generate_training_data(text, word_to_idx, window_size=5)

# 4. Реализация Word2Vec
def train_word2vec(training_data, vocab_size, embedding_dim, learning_rate=0.01, epochs=10, batch_size=1024):
    W_input = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim))  # Входной слой
    W_output = np.random.uniform(-0.5, 0.5, (embedding_dim, vocab_size))  # Выходной слой
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    n_batches = len(training_data) // batch_size
    for epoch in range(epochs):
        loss = 0
        np.random.shuffle(training_data)
        
        for batch_idx in range(n_batches):
            print(batch_idx)
            batch = training_data[batch_idx*batch_size : (batch_idx+1)*batch_size]
            target_words = [item[0] for item in batch]
            context_words = [item[1] for item in batch]
            
            # Forward pass
            hidden = W_input[target_words]  # shape: (batch_size, embedding_dim)
            output = np.dot(hidden, W_output)  # shape: (batch_size, vocab_size)
            softmax_output = softmax(output)

            # Compute loss
            loss += -np.sum(np.log(softmax_output[np.arange(batch_size), context_words]))
            
            # Backward pass
            d_output = softmax_output.copy()
            d_output[np.arange(batch_size), context_words] -= 1
            
            # Update weights
            W_output -= learning_rate * np.dot(hidden.T, d_output) / batch_size
            for i, word in enumerate(target_words):
                W_input[word] -= learning_rate * np.dot(W_output, d_output[i]) / batch_size
        
        print(f"Epoch {epoch+1}, Loss: {loss/len(training_data):.4f}")
    
    
    return W_input, W_output

# Обучение Word2Vec для каждого размера эмбеддингов
trained_models = {}
for size in params.embedding_sizes:
    print(f"Training Word2Vec with embedding dimension {size}")
    W_input, _ = train_word2vec(training_data, vocab_size, size, learning_rate=0.01, epochs=10)
    trained_models[size] = W_input
    # Сохранение
    np.save(f'output/word2vec_embeddings_{size}.npy', trained_models[size])


# 5. Подготовка данных для нейросети

def prepare_data(texts, embeddings, tokenizer, L=5):
    X, y = [], []
    for sentence in texts:
        for i in range(len(sentence) - L):
            context = sentence[i:i+L]
            target = sentence[i+L]

            # Преобразуем слова в эмбеддинги
            context_vectors = [embeddings[word_to_idx.get(word, 1)] for word in context]
            X.append(context_vectors)
            y.append(word_to_idx.get(target, 1))

    # Преобразуем целевые слова в категориальные метки
    #y_indices = [tokenizer.texts_to_sequences([word])[0][0] for word in y]
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    return np.array(X), np.array(y)


 #6. Создание и обучение нейросети
def create_model(embedding_size):
    model = Sequential([
        Input(shape=(params.L, embedding_size)),
        Flatten(),
        Dense(1500, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Обучаем модели с разными эмбедингами
history_dict = {}


# Для каждого размера эмбеддинга:
for size, embeddings in trained_models.items():
    print(f"Preparing data for embedding size {size}")
    X, y = prepare_data(text, embeddings, None, L=params.L)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"Training neural network with embedding size {size}")
    model = create_model(size)
    history = model.fit(X_train, y_train, epochs=params.epochs, batch_size=params.batch_size)
    history_dict[size] = history.history['loss']

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Size {size}: Test accuracy = {accuracy:.4f}, Test Loss = {loss:.4f}")
    model.save(f"output/model_trained{size}.keras")
    
    # предсказываем слово

    test_sentence = "В столовой с низким потолком, глубоко под землей,"
    test_tokens = preprocess_text(test_sentence)
    
    test_indices = [word_to_idx.get(word, 1) for word in test_tokens[0]]

    # Получение эмбедингов
    embeddings = trained_models[size]
    input_vector = np.array([[embeddings[word_to_idx[word]] for word in test_tokens[0]]])

    # Предсказание
    output = model.predict(input_vector)
    predicted_word = idx_to_word[np.argmax(output)]
    print(f"Predicted next word: {predicted_word}")

    

# Визуализация потерь при обучении
plt.figure(figsize=(10, 6))
for size, losses in history_dict.items():
    plt.plot(losses, label=f'Embedding size {size}')

plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
#plt.show() # просто не хочу чтоб он рисовал достал меня
plt.savefig("output/losses.jpg")


