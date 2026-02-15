# ============================================
# IMPORT LIBRARY
# ============================================

import pandas as pd
import re
import joblib

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


# ============================================
# DOWNLOAD STOPWORDS
# ============================================

nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("grab_reviews.csv", sep=";", encoding="latin1")

print("Jumlah data:", len(df))


# ============================================
# LABELING SENTIMEN
# ============================================

def label(score):

    if score <= 2:
        return "negatif"

    elif score == 3:
        return "netral"

    else:
        return "positif"


df["sentimen"] = df["score"].apply(label)


print("\nDistribusi Sentimen:")
print(df["sentimen"].value_counts())


# ============================================
# PREPROCESSING
# ============================================

def cleaning(text):

    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess(text):

    text = cleaning(text)

    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


df["clean"] = df["content"].apply(preprocess)


# ============================================
# TF-IDF (BIGRAM)
# ============================================

vectorizer = TfidfVectorizer(

    max_features=5000,
    ngram_range=(1,2)

)

X = vectorizer.fit_transform(df["clean"])

y = df["sentimen"]


# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)


print("\nJumlah data training:", len(y_train))
print("Jumlah data testing:", len(y_test))


# ============================================
# SMOTE BALANCING
# ============================================

print("\nSebelum SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nSesudah SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ============================================
# TRAIN MODEL (LinearSVC terbaik untuk teks)
# ============================================

model = LinearSVC(

    class_weight='balanced',
    max_iter=10000

)

model.fit(X_train_smote, y_train_smote)


# ============================================
# TEST MODEL
# ============================================

y_pred = model.predict(X_test)


# ============================================
# EVALUASI
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================
# SIMPAN MODEL
# ============================================

joblib.dump(model, "best_svm_model.pkl")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


print("\nModel berhasil disimpan!")
