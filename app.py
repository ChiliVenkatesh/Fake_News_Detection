import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

# ---------------------------------------------------------
# Load artifacts (must exist in the same folder as app.py)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("fake_news_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, preprocessor, feature_cols

model, preprocessor, feature_cols = load_artifacts()

text_col = feature_cols["text_col"]   # "full_text"
cat_cols = feature_cols["cat_cols"]   # ["subject"]
num_cols = feature_cols["num_cols"]   # ["text_length"]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste a news title and article text below to check if it's likely **Fake** or **True**.")

title = st.text_input("News Title")
text = st.text_area("News Text", height=250)
subject = st.selectbox(
    "Subject",
    ["Government News", "Middle-east", "News", "US_News",
     "left-news", "politics", "politicsNews", "worldnews"],
)

if st.button("Predict"):
    if not title.strip() and not text.strip():
        st.warning("Please enter a title or some text first.")
    else:
        full_text = f"{title} {text}"
        text_length = len(full_text)

        # Build a single-row dataframe matching training-time columns
        input_df = pd.DataFrame({
            text_col: [full_text],
            cat_cols[0]: [subject],
            num_cols[0]: [text_length],
        })

        X_proc = preprocessor.transform(input_df)
        X_proc = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc

        prob = float(model.predict(X_proc, verbose=0)[0][0])
        label = "TRUE" if prob > 0.5 else "FAKE"
        confidence = prob if label == "TRUE" else 1 - prob

        st.subheader("Result")
        if label == "TRUE":
            st.success(f"✅ Likely **TRUE** news ({confidence:.1%} confidence)")
        else:
            st.error(f"🚫 Likely **FAKE** news ({confidence:.1%} confidence)")

        st.caption(f"Raw model output (P(true)): {prob:.4f}")