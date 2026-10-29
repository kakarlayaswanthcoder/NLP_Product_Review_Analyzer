import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")



stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
   
    text = text.lower()

    tokens = word_tokenize(text)

    
    tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]

    
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

   
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return tokens