import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("🛒 Online Purchase Prediction")

st.write("Enter user session details:")

page_values = st.number_input("Page Value")
bounce_rates = st.number_input("Bounce Rate")
exit_rates = st.number_input("Exit Rate")
product_related = st.number_input("Product Related Pages")

visitor_type = st.selectbox("Visitor Type", ["Returning", "New"])
weekend = st.selectbox("Weekend", ["No", "Yes"])

# Encoding
visitor_type = 1 if visitor_type == "Returning" else 0
weekend = 1 if weekend == "Yes" else 0

features = np.array([[page_values, bounce_rates, exit_rates, product_related, visitor_type, weekend]])

if st.button("Predict"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ User will Purchase")
    else:
        st.error("❌ No Purchase")