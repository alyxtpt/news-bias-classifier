import streamlit as st
from pathlib import Path
import joblib

MODEL_PATH = Path("models/model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def main():
    st.title("News Bias Classifier")
    st.write("Paste a news article and the model will try to classify the bias in the text")

    article_text = st.text_area("News article", height = 300)

    if st.button("Classify") and article_text.strip():
        model = load_model()
        label = model.predict([article_text])[0]
        probability = model.predict_proba([article_text])[0]
        classes = model.classes_

        st.subheader("Guessed bias:", label)
        st.write("Probabilities:")
        for i, j in zip(classes, probability):
            st.write(f"- **{i}**: {j:.3f}")
        
if __name__ == "__main__":
    main()