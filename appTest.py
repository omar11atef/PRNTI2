import streamlit as st
import pandas as pd
import joblib
import gdown
import tensorflow as tf
import os

# === روابط الملفات على Google Drive ===
files = {
    "best_ml_model.pkl": "https://drive.google.com/uc?id=1Hrj1_EKfwqozCTM0FaVi82Qg1nnyfThn",
    "preprocessor.pkl": "https://drive.google.com/uc?id=1ZyiR3ZiGNXzDihWBuTTIK8PnTa-C0Ew_",
    "nn_model.h5": "https://drive.google.com/uc?id=13eoCKB9sk3JqPq0qIynO05E_m1y5ebTU",
    "zomato_sample.csv": "https://drive.google.com/uc?id=1U3CMhKvQ2_lOaFKVpFQVDxrr6hV4UE_Z"
}

# === تحميل الملفات إذا لم تكن موجودة ===
for filename, url in files.items():
    if not os.path.exists(filename):
        st.write(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)

# === تحميل النماذج والمعالج ===
preprocessor = joblib.load("preprocessor.pkl")
best_model = joblib.load("best_ml_model.pkl")
nn_model = tf.keras.models.load_model("nn_model.h5")

# === واجهة Streamlit ===
st.title("🍽 Zomato Restaurant Prediction App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset")
    data = pd.read_csv("zomato_sample.csv")

st.write("### Preview Data", data.head())

# === تجهيز البيانات ===
processed_data = preprocessor.transform(data)

# === توقعات النماذج ===
ml_preds = best_model.predict(processed_data)
nn_preds = nn_model.predict(processed_data)

st.write("### ML Model Predictions", ml_preds)
st.write("### NN Model Predictions", nn_preds)
