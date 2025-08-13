import streamlit as st
import pandas as pd
import joblib
import gdown
import tensorflow as tf
import os

# Google Drive direct download links
files = {
    "best_ml_model.pkl": "https://drive.google.com/uc?id=1Hrj1_EKfwqozCTM0FaVi82Qg1nnyfThn",
    "preprocessor.pkl": "https://drive.google.com/uc?id=1ZyiR3ZiGNXzDihWBuTTIK8PnTa-C0Ew_",
    "nn_model.h5": "https://drive.google.com/uc?id=13eoCKB9sk3JqPq0qIynO05E_m1y5ebTU",
    "zomato_sample.csv": "https://drive.google.com/uc?id=1U3CMhKvQ2_lOaFKVpFQVDxrr6hV4UE_Z"
}

# Download files if not exist
for filename, url in files.items():
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

# Load dataset
df = pd.read_csv("zomato_sample.csv")

# Load models
preprocessor = joblib.load("preprocessor.pkl")
best_model = joblib.load("best_ml_model.pkl")
nn_model = tf.keras.models.load_model("nn_model.h5")

# Streamlit pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analysis", "Prediction"])

if page == "Analysis":
    st.title("Zomato Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write("### Dataset Info")
    st.write(df.describe())

elif page == "Prediction":
    st.title("Restaurant Prediction")
    online_order = st.selectbox("Online Order", ["Yes", "No"])
    book_table = st.selectbox("Book Table", ["Yes", "No"])
    votes = st.number_input("Votes", min_value=0)
    cost = st.number_input("Approx Cost for Two People", min_value=0)
    location = st.text_input("Location")
    rest_type = st.text_input("Restaurant Type")
    listed_type = st.text_input("Listed In Type")
    listed_city = st.text_input("Listed In City")

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "online_order": [online_order],
            "book_table": [book_table],
            "votes": [votes],
            "approx_cost(for_two_people)": [cost],
            "location": [location],
            "rest_type": [rest_type],
            "listed_in(type)": [listed_type],
            "listed_in(city)": [listed_city]
        })

        processed_input = preprocessor.transform(input_df)
        ml_pred = best_model.predict(processed_input)
        nn_pred = nn_model.predict(processed_input)

        st.write(f"**ML Model Prediction:** {ml_pred[0]}")
        st.write(f"**Neural Network Prediction:** {nn_pred[0][0]}")
