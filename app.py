import streamlit as st
import pickle
import numpy as np


st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")

st.title("🔗 Hierarchical Clustering Prediction")
st.write("Enter feature values to find the cluster")


def load_model():
    with open("Hierarchical_Cluster_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()


feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

input_data = np.array([[feature1, feature2]])

if st.button("Predict Cluster"):
    try:
        # Some hierarchical models use fit_predict
        if hasattr(model, "predict"):
            cluster = model.predict(input_data)
        else:
            cluster = model.fit_predict(input_data)

        st.success(f"✅ Predicted Cluster: {cluster[0]}")
    except Exception as e:
        st.error("⚠️ Prediction failed")
        st.write(str(e))
