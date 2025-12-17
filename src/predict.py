import joblib
from pathlib import Path

MODEL_PATH = Path("model/model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def predict_bias(article_text: str) -> str:
    model = load_model()
    label = model.predict([article_text])[0]
    probability = model.predict_proba([article_text])[0]
    classes = model.classes_
    return label, dict(zip(classes, probability))

if __name__ == "__main__":
    text = input("Paste article text:\n")
    label, probs = predict_bias(text)
    print("Guessed bias:", label)
    print("Probability:")
    for i, j in probs.items():
        print(f"    {i}: {j:.3f}")