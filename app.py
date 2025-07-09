import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import joblib

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



mnb = joblib.load('mnb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# tfidf = pickle.load(tfidf,open('vectorizer.pkl','rb'))
# model = pickle.load(mnb,open('model.pkl','rb'))


st.markdown("""
    <style>
    /* Set background color */
    .stApp {
        background-color: #eef2f7;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Custom title */
    .title {
        font-size: 40px;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
    }

    /* Text area styling */
    textarea {
        border: 2px solid #2c3e50 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 30px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: black;
    }

    /* Result text styling */
    .stSuccess, .stError {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📧 Email Spam Classifier</div>', unsafe_allow_html=True)




# st.title("Email/SMS Spam Detector")

input_sms = st.text_area("Enter the message")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = vectorizer.transform([transformed_sms])
    # 3. predict
    result = mnb.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")