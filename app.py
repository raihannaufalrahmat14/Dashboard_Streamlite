import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. NLTK Data Downloads (Mencegah LookupError) ---
def download_nltk_resources():
    resources = {
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/stopwords': 'stopwords'
    }
    for path, resource in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# --- 2. Global Initializations ---
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- 3. Preprocessing Functions ---
def cleaning(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    cleaned_text = cleaning(text)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# --- 4. Load Model and TF-IDF Vectorizer ---
@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load('best_svm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: File model (.pkl) tidak ditemukan. Pastikan file ada di folder yang sama.")
        st.stop()

best_svm_model, tfidf_vectorizer = load_ml_models()

# --- 5. Load Data & Evaluation ---
@st.cache_data
def load_and_evaluate():
    try:
        df = pd.read_csv('grab_reviews.csv', sep=';', encoding='latin1')
    except FileNotFoundError:
        st.error("Error: 'grab_reviews.csv' tidak ditemukan.")
        st.stop()

    # Preprocessing DataFrame
    df['final_text'] = df['content'].apply(cleaning).apply(preprocess_text)
    
    def label_sentiment(score):
        if score <= 2: return 'negatif'
        elif score == 3: return 'netral'
        else: return 'positif'
    
    df['sentimen'] = df['score'].apply(label_sentiment)

    # Menghitung Distribusi untuk Plot
    temp_df = df.groupby('sentimen').count()['content'].reset_index()
    temp_df.columns = ['Sentimen', 'Jumlah']
    temp_df['Percentage'] = (temp_df['Jumlah'] / temp_df['Jumlah'].sum()) * 100

    # Split Data & Hitung Confusion Matrix
    X = df['final_text']
    y = df['sentimen']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = best_svm_model.predict(X_test_tfidf)
    
    cm = confusion_matrix(y_test, y_pred, labels=['negatif', 'netral', 'positif'])
    report = classification_report(y_test, y_pred, output_dict=True)

    # Siapkan Teks Wordcloud
    pos = " ".join(df[df['sentimen'] == 'positif']['final_text'])
    neu = " ".join(df[df['sentimen'] == 'netral']['final_text'])
    neg = " ".join(df[df['sentimen'] == 'negatif']['final_text'])

    return temp_df, pos, neu, neg, cm, report

temp_df, positive_text, neutral_text, negative_text, cm, report = load_and_evaluate()

# --- 6. Streamlit UI ---
st.set_page_config(page_title="Grab Sentiment Analysis", layout="wide")
st.title("Aplikasi Analisis Sentimen Ulasan Grab")

# Section: Prediksi Tunggal
st.header("Prediksi Sentimen Teks")
user_input = st.text_area("Masukkan ulasan:", "Aplikasi bagus dan sangat membantu.")
if st.button("Analisis"):
    processed = preprocess_text(user_input)
    vec = tfidf_vectorizer.transform([processed])
    res = best_svm_model.predict(vec)
    st.success(f"Hasil Prediksi: **{res[0].upper()}**")

st.divider()

# Section: Visualisasi Distribusi
st.header("Visualisasi Distribusi Sentimen")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bar Chart")
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(data=temp_df, x='Sentimen', y='Jumlah', palette='viridis', ax=ax_bar)
    st.pyplot(fig_bar)

with col2:
    st.subheader("Donut Chart")
    fig_donut, ax_donut = plt.subplots(figsize=(6,6))
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    ax_donut.pie(temp_df['Percentage'], labels=temp_df['Sentimen'], autopct='%1.1f%%', 
                 startangle=90, colors=colors, wedgeprops=dict(width=0.3))
    st.pyplot(fig_donut)

st.divider()

# Section: Confusion Matrix
st.header("Evaluasi Model: Confusion Matrix")
col_cm, col_met = st.columns([2, 1])

with col_cm:
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['negatif', 'netral', 'positif'],
                yticklabels=['negatif', 'netral', 'positif'])
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    st.pyplot(fig_cm)

with col_met:
    st.metric("Akurasi Model", f"{report['accuracy']:.2%}")
    st.dataframe(pd.DataFrame(report).transpose().iloc[:3, :3])

st.divider()

# Section: WordCloud
st.header("Word Clouds")
wc_cols = st.columns(3)
titles = ["Positif", "Netral", "Negatif"]
texts = [positive_text, neutral_text, negative_text]

for i, col in enumerate(wc_cols):
    with col:
        st.subheader(titles[i])
        if texts[i].strip():
            wc = WordCloud(background_color='white').generate(texts[i])
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Data tidak cukup.")
