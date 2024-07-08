import streamlit as st
import openai
import numpy as np
import torch
import traceback

st.title("Text Similarity Checker")

# Retrieve OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Calculate"):
    try:
        # Function to get embeddings from OpenAI
        def get_embedding(text):
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"  # Change to the desired model
            )
            if 'data' in response and response['data'] and 'embedding' in response['data'][0]:
                return np.array(response['data'][0]['embedding'])
            else:
                raise ValueError("Invalid response format from OpenAI API")

        # Calculate embeddings
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)

        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            vec1 = torch.tensor(vec1)
            vec2 = torch.tensor(vec2)
            dot_product = torch.dot(vec1, vec2).item()
            magnitude = torch.norm(vec1) * torch.norm(vec2)
            return dot_product / magnitude if magnitude != 0 else 0

        similarity = cosine_similarity(embedding1, embedding2)

        # Define your threshold
        SIMILARITY_THRESHOLD = 0.8

        if similarity > SIMILARITY_THRESHOLD:
            st.write("true")
        else:
            st.write("false")

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API request failed: {e}")
        traceback.print_exc()  # Print traceback for more detailed error information
    except ValueError as ve:
        st.error(f"Error processing OpenAI response: {ve}")
        traceback.print_exc()  # Print traceback for more detailed error information
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        traceback.print_exc()  # Print traceback for more detailed error information
