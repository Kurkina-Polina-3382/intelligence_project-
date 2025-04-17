
import nltk
import re 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from joblib import Parallel, delayed

# Скачиваем необходимые ресурсы NLTK
nltk.download('stopwords')
nltk.download('punkt')
# Инициализация объектов 1 раз (вместо создания в каждой итерации)
stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words('russian'))
regex_pattern = re.compile(r'[^а-яёa-z\s]')

def process_sentence(sent):
    """Обработка одного предложения"""
    sent = sent.lower()
    sent = regex_pattern.sub('', sent)
    tokens = word_tokenize(sent, language='russian')
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens if tokens else None

def preprocess_text(text, n_jobs=4):
    """Оптимизированный пайплайн с параллельной обработкой"""
    sentences = sent_tokenize(text, language='russian')
    
    # Параллельная обработка
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sentence)(sent) 
        for sent in sentences
    )
    
    # Фильтрация None и пустых списков
    return [tokens for tokens in results if tokens]

