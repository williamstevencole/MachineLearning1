import nltk
import re
import random
import pandas as pd

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

TEST_SIZE = 0.2
RANDOM_STATE = 42
stop_words = None

pd.set_option('display.max_colwidth', 100)

def load_data() -> pd.DataFrame:
    nltk.download('movie_reviews') # Corpus

    nltk.download('punkt') # For tokenization
    nltk.download('punkt_tab') # For tokenization

    nltk.download('stopwords')

    docs = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()  # 'pos' y 'neg'
        for fileid in movie_reviews.fileids(category)
    ]

    random.shuffle(docs)

    df = pd.DataFrame(docs, columns=["text", "label"])
    return df


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        w for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


def split_train_test(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> tuple:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def create_vectorizer(X_train) -> tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1) # 1 word (unigram) 
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec


def train_model(X_train_vec, y_train) -> MultinomialNB:
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model


def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print("=" * 40, " Resultados de la evaluación ", "=" * 40)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()


def run_examples(model, vectorizer):
    ejemplos = [
      "I absolutely loved this movie, it was fantastic and brilliant!",
      "This was the worst film I have ever seen, completely boring.",
      "The movie was okay, not great, but not terrible either.",
      "Amazing acting but the story was very weak and predictable.",
      "A beautiful soundtrack and stunning visuals made it unforgettable.",
      "I couldn't get through it; the plot made no sense.",
      "Performances were solid but pacing dragged in the middle.",
      "An inspiring film that left me feeling uplifted.",
      "Predictable twists and flat dialogue ruined the experience.",
      "A charming indie with lots of heart and clever humor.",
      "Too long and self-indulgent despite some good scenes.",
      "Clever script and great chemistry between the leads.",
      "Mediocre effects and a thin storyline.",
      "A surprisingly powerful ending that made the film worthwhile.",
      "Average at best; there are better movies in this genre.",
      "Hilarious from start to finish, I laughed out loud.",
      "The cinematography was exceptional but the themes were shallow.",
      "Left me cold — no emotional connection to the characters.",
      "A masterpiece of modern cinema, brilliantly directed.",
      "Good intentions but poor execution.",
      "Fast-paced thriller with non-stop action and suspense.",
      "I found the dialogue cringeworthy and unnatural.",
      "A heartfelt story that resonated with me on many levels.",
      "Not my cup of tea; I got bored halfway through."
    ]

    ejemplos_limpios = [preprocess(t) for t in ejemplos]
    ejemplos_vec = vectorizer.transform(ejemplos_limpios)
    predicciones = model.predict(ejemplos_vec)

    for texto, etiqueta in zip(ejemplos, predicciones):
        print(f"Texto: {texto}")
        print(f"Predicción de sentimiento: {etiqueta}")
        print("-" * 50)


def main():
    global stop_words

    df = load_data()
    print(df.head(), "\n")

    stop_words = set(stopwords.words('english'))

    print("\n"*4)
    print("PRE PREPROCESSING:")
    print(df[["label", "text"]].head(10), "\n")

    df["clean_text"] = df["text"].apply(preprocess)

    print("POST PREPROCESSING:")
    print(df[["label", "clean_text"]].head(10), "\n")
    print("\n"*4)

    X = df["clean_text"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = split_train_test(X, y)

    vectorizer, X_train_vec = create_vectorizer(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    model = train_model(X_train_vec, y_train)
    evaluate_model(model, X_test_vec, y_test)
    run_examples(model, vectorizer)

if __name__ == "__main__":
    main()
