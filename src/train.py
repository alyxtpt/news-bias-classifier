import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/articles.csv")
MODEL_PATH = Path("models/model.joblib")

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df["text"], df["label"]

def train():
    x, y, = load_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, stratify = y)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = 10000,
            ngram_range = (1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter = 1000))
    ])

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    train()