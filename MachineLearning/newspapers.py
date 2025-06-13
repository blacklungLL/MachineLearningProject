import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# def load_data_from_folders(base_path):
#     data = []
#     categories = os.listdir(base_path)  # Получаем список папок (категорий)
#
#     for category in categories:
#         category_path = os.path.join(base_path, category)
#         if os.path.isdir(category_path):  # Проверяем, что это папка
#             for filename in os.listdir(category_path):
#                 file_path = os.path.join(category_path, filename)
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                     text = file.read()
#                     data.append({'text': text, 'category': category})
#
#     return pd.DataFrame(data)
#
#
# base_path = "bbc"
# data = load_data_from_folders(base_path)
#
#
# print(data)
#
# data.to_csv("news_dataset.csv", index=False)

data = pd.read_csv("news_dataset.csv")

print(data.head())

# Предобработка текста
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Токенизация
    tokens = word_tokenize(text.lower(), language='english')  # Явно указываем язык
    # Удаление стоп-слов и пунктуации
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# Применение функции предобработки
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Кодирование меток
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'],
    data['category_encoded'],
    test_size=0.2,
    random_state=42
)


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Ограничим количество признаков до 5000
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Обучение логистической регрессии
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Предсказания
y_pred_lr = lr_model.predict(X_test_tfidf)

# Оценка модели
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

# Обучение случайного леса
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Предсказания
y_pred_rf = rf_model.predict(X_test_tfidf)

# Оценка модели
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Визуализация распределения категорий
plt.figure(figsize=(8, 6))
sns.countplot(x='category', data=data)
plt.title("Распределение категорий")
plt.show()

# Матрица ошибок
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Матрица ошибок (Logistic Regression)")
plt.xlabel("Предсказано")
plt.ylabel("Фактическое")
plt.show()