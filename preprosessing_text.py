
import nltk
import re 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer

# Скачиваем необходимые ресурсы NLTK
nltk.download('stopwords')
nltk.download('punkt')
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

