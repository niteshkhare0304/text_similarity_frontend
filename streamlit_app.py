# text_similarity_project/streamlit_app.py

import streamlit as st
import requests

st.title("Text Similarity Checker")

text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Calculate"):
    try:
        response = requests.post("http://localhost:8000/compare_texts", json={"text1": text1, "text2": text2})
        response.raise_for_status()  # Raise an error for bad status codes

        result = response.json()
        st.write(f"Similarity result: {result}")

        if result:  # If result is true
            st.success("The texts are similar.")
        else:  # If result is false
            st.error("The texts are not similar.")

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

    except ValueError:
        st.error("Invalid response from API")
