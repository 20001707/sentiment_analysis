import streamlit as st
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
import nltk
import pickle

nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')

with open("models/count_vectorizer.pickle",'rb') as f:
    cv = pickle.load(f)
f.close()

with open("models/lr_model.pickle",'rb') as f:
    model = pickle.load(f)
f.close()


def clean_text(text):
    def remove_mentions(text):
        # Regular expression pattern to match mentions
        mention_pattern = r'@[\w_]+'

        # Remove mentions using regular expression substitution
        cleaned_text = re.sub(mention_pattern, '', text)

        return cleaned_text

    # Remove mentions from the text
    text = remove_mentions(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Handling contractions
    contractions = {
        "n't": "not",
        "'s": "is",
        "'re": "are",
        "'ve": "have"

    }
    tokens = [contractions[token] if token in contractions else token for token in tokens]

    # Removing stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


st.title("Sentimental Analysis")
input_text = st.text_area(height=80, label='Enter the text to analyse')

sentiments = ['Negative', 'Neutral', 'Positive']

def predict(text):
    cleaned_sample_text = ' '.join(clean_text(text))
    result = model.predict(cv.transform([cleaned_sample_text]).toarray())[0]
    return sentiments[result]

if st.button("Analyze"):
    output = "The predicted sentiment is : " + str(predict(input_text))
    st.write(output)