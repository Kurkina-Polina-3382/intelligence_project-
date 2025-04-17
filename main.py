import numpy as np   
import re               
import os
import matplotlib.pyplot as plt       
   
import pickle       
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer

# Скачиваем необходимые ресурсы NLTK
nltk.download('stopwords')
nltk.download('punkt')

import params as params

from gensim.models import Word2Vec


def preprocess_text(text):
    
    data = []  

    stemmer = SnowballStemmer("russian")
    stop_words = set(stopwords.words('russian'))

    sentences = sent_tokenize(text, language='russian')  # преобразуем текст в список предложений (токенов)
    
    for sent in sentences:
        # Приведение к нижнему регистру
        sent = sent.lower()
            
        # Удаление спецсимволов и цифр
        sent = re.sub(r'[^а-яёa-z\s]', '', sent) 
        
        
        # Токенизация
        tokens = word_tokenize(sent, language='russian')


        # Удаление стоп-слов
        tokens = [word for word in tokens if word not in stop_words]
        
        # Стемминг
        # но вообще можно использовать лемматизацию. она должна давать лучше результат (бежал - бегать) morph = MorphAnalyzer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        data.append(tokens)  # Добавляем список токенов (предложение)
        #может и не делать разбиение на предложение
                
    # Фильтрация от пустых предложений
    data = [sent for sent in data if sent]
    return data

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

# Преобразование в числовые индексы Векторизация (токениза
# ция в числа)
tokenizer = Tokenizer(num_words=10_000, oov_token=0)
tokenizer.fit_on_texts(text)

with open('output/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# эмбединги
word2vec_models = {}

for size in params.embedding_sizes:
    model = Word2Vec(text, vector_size=size, window=5, min_count=1, workers=4)
    word2vec_models[size] = model
    # сохраняем
    model.save(f"output/word2vec{size}.model")

# 3. Подготовка данных для нейросети

def prepare_data(texts, word2vec_model, tokenizer, L=5):
    X, y = [], []
    for sentence in texts:
        for i in range(len(sentence) - L):
            # Берем L слов и следующее за ними
            context = sentence[i:i+L]
            target = sentence[i+L]
            
            # Заменяем слова на векторы
            context_vectors = [word2vec_model.wv[word] for word in context]
            X.append(context_vectors)
            y.append(target)
    # Преобразуем  ожидаемые слова в индексы через tokenizer
    y_indices = tokenizer.texts_to_sequences(y)
    y = tf.keras.utils.to_categorical(y_indices, num_classes=params.vocab_size)
    
    return np.array(X), np.array(y)


# 4. Создание и обучение нейросети
def create_model(embedding_size):
    
    model = Sequential([
        Input(shape=(5, embedding_size)),
        #  про Flatten Выравнивает входные данные. Не влияет на размер пакета.
        # Примечание: если входные данные имеют форму (batch,) без оси признаков, то при сглаживании добавляется дополнительный размер канала, и выходная форма будет (batch, 1).
        Flatten(),
        Dense(1500, activation='relu'),
        Dense(params.vocab_size, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Обучаем модели с разными эмбедингами
history_dict = {}


# Для каждого размера эмбеддинга:
for size in params.embedding_sizes:
    X, y = prepare_data(text, word2vec_models[size], tokenizer, L=params.L)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = create_model(size)
    history = model.fit(X_train, y_train, epochs=params.epochs, batch_size=params.batch_size)
    history_dict[size] = history.history['loss']
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Size {size}: Test accuracy = {accuracy:.4f} Test Loss: {loss:.4f}")
    # Сохраняем модель в файл
    model.save(f"output/model_trained{size}.keras")  
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


