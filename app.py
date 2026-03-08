import streamlit as st
import pickle

# load files
tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message")

    else:
        vector_input = tfidf.transform([input_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam 🚨")
        else:
            st.header("Not Spam ✅")
