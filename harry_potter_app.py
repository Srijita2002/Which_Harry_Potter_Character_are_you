import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load the dataset of Harry Potter character traits
data = pd.read_csv('harry_potter_characters.csv')

# Preprocess the dataset
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Traits'].values.astype('U'))

# Function to determine the character based on user input
def get_harry_potter_character(user_input):
    # Transform user input into a feature vector
    user_vector = tfidf.transform([user_input])

    # Calculate cosine similarity between user input and character traits
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    # Find the character with the highest similarity score
    character_index = similarity_scores.argmax()
    character = data['Character'][character_index]

    return character

# Streamlit application
def main():
    st.set_page_config(page_title="Harry Potter Character Quiz", page_icon="🧙‍♂️")

    # Background image
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://preview.redd.it/cxeimwflyo881.png?width=800&format=png&auto=webp&s=2160be26be2b962a1b9048c2e6236bbfe643c71c");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Title
    st.markdown("<h1 style='text-align: center; color: yellow;'>Which Harry Potter Character Are You?</h1>", unsafe_allow_html=True)

    # User input
    st.markdown("<p style='color: orange;'>Enter your personality traits (separated by spaces):</p>", unsafe_allow_html=True)
    user_input = st.text_area("", "")
    if st.button("Find Out"):
        if user_input:
            character = get_harry_potter_character(user_input)
            st.markdown(f"<h2 style='text-align: center; color: pink;'>You are: {character}</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: pink;'>Please enter your personality traits</h2>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
