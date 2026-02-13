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

# --- NLTK Data Downloads (Perbaikan Error) ---
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

# --- Global Initializations ---
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Preprocessing Functions ---
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

# --- Load Model and TF-IDF Vectorizer ---
@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load('best_svm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or TF-IDF vectorizer files not found. Please ensure 'best_svm_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()

best_svm_model, tfidf_vectorizer = load_ml_models()

# --- Load Data, Prepare Visualizations, and Evaluate ---
@st.cache_data
def load_data_and_prepare():
    try:
        df_original = pd.read_csv('grab_reviews.csv', sep=';', encoding='latin1')
    except FileNotFoundError:
        st.error("Error: 'grab_reviews.csv' not found. Please upload the dataset.")
        st.stop()

    df_vis = df_original.copy()

    # Preprocessing
    df_vis['clean_text'] = df_vis['content'].apply(cleaning)
    df_vis['token'] = df_vis['clean_text'].apply(word_tokenize)
    df_vis['stopword_removed'] = df_vis['token'].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    df_vis['stemmed'] = df_vis['stopword_removed'].apply(
        lambda x: [stemmer.stem(word) for word in x]
    )
    df_vis['final_text'] = df_vis['stemmed'].apply(lambda x: ' '.join(x))

    def label_sentiment_vis(score):
        if score <= 2:
            return 'negatif'
        elif score == 3:
            return 'netral'
        else:
            return 'positif'

    df_vis['sentimen'] = df_vis['score'].apply(label_sentiment_vis)

    # --- Data untuk Plot Distribusi ---
    temp_df = df_vis.groupby('sentimen').count()['content'].reset_index().sort_values(by='content', ascending=False)
    temp_df.columns = ['Sentimen', 'Jumlah Prediksi']
    temp_df['Percentage'] = (temp_df['Jumlah Prediksi'] / temp_df['Jumlah Prediksi'].sum()) * 100

    # --- Data untuk Word Cloud ---
    positive_text = " ".join(df_vis[df_vis['sentimen'] == 'positif']['final_text'])
    neutral_text = " ".join(df_vis[df_vis['sentimen'] == 'netral']['final_text'])
    negative_text = " ".join(df_vis[df_vis['sentimen'] == 'negatif']['final_text'])

    # --- Split Data & Confusion Matrix ---
    X = df_vis['final_text']
    y = df_vis['sentimen']
    
    # Split data: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transformasi dan Prediksi Data Uji
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = best_svm_model.predict(X_test_tfidf)
    
    # Hitung Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['negatif', 'netral', 'positif'])
    report = classification_report(y_test, y_pred, output_dict=True)

    return temp_df, positive_text, neutral_text, negative_text, cm, report

# Panggil fungsi yang sudah diupdate
temp_df, positive_text, neutral_text, negative_text, cm, report = load_data_and_prepare()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Aplikasi Analisis Sentimen Ulasan Grab")
st.markdown("---")

# --- Sentiment Prediction Section ---
st.header("Prediksi Sentimen Teks")
user_input = st.text_area("Masukkan teks ulasan Anda di sini:", "Aplikasi ini sangat membantu dan cepat.")

if st.button("Prediksi Sentimen"):
    if user_input:
        processed_input = preprocess_text(user_input)
        vectorized_input = tfidf_vectorizer.transform([processed_input])
        prediction = best_svm_model.predict(vectorized_input)
        st.success(f"Prediksi Sentimen: **{prediction[0].upper()}**")
    else:
        st.warning("Mohon masukkan teks untuk prediksi.")

st.markdown("---<br>", unsafe_allow_html=True)

# --- Visualization Section ---
st.header("Visualisasi Distribusi Sentimen")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Sentimen (Bar Plot)")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    sns.barplot(x=temp_df['Sentimen'], y=temp_df['Jumlah Prediksi'], palette='viridis', ax=ax_bar)
    ax_bar.set_title('Distribusi Sentimen Ulasan')
    ax_bar.set_xlabel('Sentimen')
    ax_bar.set_ylabel('Jumlah Ulasan')
    st.pyplot(fig_bar)

with col2:
    st.subheader("Distribusi Sentimen (Donut Plot)")
    fig_donut, ax_donut = plt.subplots(figsize=(8, 8))
    labels = temp_df['Sentimen']
    sizes = temp_df['Percentage']
    colors = {'negatif': '#FF9999', 'netral': '#66B2FF', 'positif': '#99FF99'}
    pie_colors = [colors[label] for label in labels]

    wedges, texts, autotexts = ax_donut.pie(sizes, colors=pie_colors, autopct='%1.1f%%', startangle=90,
                                            pctdistance=0.85, wedgeprops=dict(width=0.3, edgecolor='white'))

    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig_donut.gca().add_artist(centre_circle)

    ax_donut.axis('equal')
    ax_donut.set_title('Distribusi Sentimen Ulasan (Donut Plot)')
    legend_labels = [f'{l} ({s:.1f}%)' for l, s in zip(labels, sizes)]
    ax_donut.legend(wedges, legend_labels, title="Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)
    for text in texts:
        text.set_fontsize(12)
    st.pyplot(fig_donut)

st.markdown("---<br>", unsafe_allow_html=True)

# --- Evaluation Section (Confusion Matrix) ---
st.header("Evaluasi Model: Confusion Matrix")


col_cm, col_rep = st.columns([2, 1])

with col_cm:
    st.subheader("Heatmap Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negatif', 'netral', 'positif'],
                yticklabels=['negatif', 'netral', 'positif'], ax=ax_cm)
    ax_cm.set_xlabel('Prediksi')
    ax_cm.set_ylabel('Aktual')
    st.pyplot(fig_cm)

with col_rep:
    st.subheader("Metrik Performa")
    accuracy = report['accuracy']
    st.metric("Overall Accuracy", f"{accuracy:.2%}")
    
    # Menampilkan tabel presisi/recall sederhana
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.iloc[:3, :3]) # Menampilkan baris negatif, netral, positif

st.markdown("---<br>", unsafe_allow_html=True)

# --- Word Cloud Section ---
st.header("Word Clouds Berdasarkan Sentimen")

wc_col1, wc_col2, wc_col3 = st.columns(3)

# Word Cloud - Positive
with wc_col1:
    st.subheader("Positif")
    if positive_text:
        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
        ax_pos.imshow(wordcloud_positive, interpolation='bilinear')
        ax_pos.axis('off')
        st.pyplot(fig_pos)
    else:
        st.info("Tidak ada data positif untuk word cloud.")

# Word Cloud - Neutral
with wc_col2:
    st.subheader("Netral")
    if neutral_text:
        wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_text)
        fig_neu, ax_neu = plt.subplots(figsize=(10, 5))
        ax_neu.imshow(wordcloud_neutral, interpolation='bilinear')
        ax_neu.axis('off')
        st.pyplot(fig_neu)
    else:
        st.info("Tidak ada data netral untuk word cloud.")

# Word Cloud - Negative
with wc_col3:
    st.subheader("Negatif")
    if negative_text:
        wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
        ax_neg.imshow(wordcloud_negative, interpolation='bilinear')
        ax_neg.axis('off')
        st.pyplot(fig_neg)
    else:
        st.info("Tidak ada data negatif untuk word cloud.")
