import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Предобработка текста
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text.lower(), language='english')
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


data = pd.read_csv("balanced_news_dataset.csv")
data['cleaned_text'] = data['text'].apply(preprocess_text)

categories = data['category'].unique()
balanced_data = []

# from sklearn.utils import resample
# min_samples = data['category'].value_counts().min()
#
# for category in categories:
#     subset = data[data['category'] == category]
#     balanced_subset = resample(subset, replace=False, n_samples=min_samples, random_state=42)
#     balanced_data.append(balanced_subset)
#
#
# balanced_data = pd.concat(balanced_data)
#
#
# print(balanced_data['category'].value_counts())
#
# balanced_data.to_csv("balanced_news_dataset.csv", index=False)

label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['cleaned_text'])
y = data['category_encoded']

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_tfidf, y)


# Функция для классификации текста
def classify_text():
    user_input = text_entry.get("1.0", "end-1c").strip()
    if not user_input:
        messagebox.showwarning("Предупреждение", "Пожалуйста, введите текст.")
        return

    cleaned_text = preprocess_text(user_input)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = lr_model.predict(vectorized_text)[0]
    category = label_encoder.inverse_transform([prediction])[0]

    result_label.config(text=f"Категория: {category}", font=("Times New Roman", 16, "bold"), fg="green")


root = tk.Tk()
root.title("Классификатор новостей")
root.geometry("800x700")
root.configure(bg="#f0f0f0")


title_label = tk.Label(
    root,
    text="Классификатор новостей",
    font=("Times New Roman", 20, "bold"),
    bg="#f0f0f0",
    fg="#333333"
)
title_label.pack(pady=10)

categories_for_window = ", ".join(label_encoder.classes_)
description_label = tk.Label(
    root,
    text=f"Данный классификатор может отнести описанную вами новость\n"
         f"в одну из следующих категорий: {categories_for_window}.",
    font=("Times New Roman", 16),
    bg="#f0f0f0",
    fg="#555555",
    wraplength=700
)
description_label.pack(pady=5)

language_label = tk.Label(
    root,
    text="Убедительная просьба писать на английском, так как модель была обучена именно на нем! Спасибо за понимание",
    font=("Times New Roman", 16),
    bg="#f0f0f0",
    fg="#555555",
    wraplength=500
)
language_label.pack(pady=10)

text_frame = tk.Frame(root, bg="#f0f0f0")
text_frame.pack(pady=10)

scrollbar = tk.Scrollbar(text_frame, orient="vertical")
scrollbar.pack(side="right", fill="y")

text_entry = tk.Text(
    text_frame,
    height=20,
    width=70,
    font=("Times New Roman", 14),
    wrap="word",
    bg="white",
    fg="#333333",
    relief="solid",
    borderwidth=1,
    insertwidth=2,
    insertbackground="black",
    yscrollcommand=scrollbar.set
)
text_entry.pack(side="left", fill="both", expand=True)

scrollbar.config(command=text_entry.yview)

classify_button = tk.Button(
    root,
    height=2,
    width=15,
    text="Классифицировать",
    command=classify_text,
    font=("Arial", 12)
)
classify_button.pack(pady=10)

result_label = tk.Label(root, text="Категория: ", font=("Times New Roman", 20, "bold"), bg="#f0f0f0", fg="#333333")
result_label.pack(pady=20)

root.mainloop()
