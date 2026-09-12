import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume Job Role Recommender",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder


# Load everything
model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Replace new lines, tabs and carriage returns
    text = re.sub(r'[\n\t\r]+', ' ', text)

    # Keep letters, numbers, +, #, ., - and spaces
    text = re.sub(r'[^a-zA-Z0-9+#.\- ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# HEADER REMOVAL
# ============================================================

def remove_header(text):

    text = re.sub(r'^.*?\s-\s', '', text, count=1)

    return text.strip()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_job_roles(resume_text):

    # Step 1: Clean text
    cleaned_text = clean_text(resume_text)

    # Step 2: Remove header
    cleaned_text = remove_header(cleaned_text)

    # Step 3: Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Step 4: Pad sequence to 500 tokens
    padded_sequence = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=500,
        padding="pre",
        truncating="post"
    )

    # Step 5: Make prediction
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Step 6: Get top 5 class indices
    top_5_indices = np.argsort(probabilities)[::-1][:5]

    # Step 7: Convert class indices to job names
    results = []

    for index in top_5_indices:

        job_role = label_encoder.inverse_transform([index])[0]

        probability = probabilities[index] * 100

        results.append(
            (job_role, probability)
        )

    return results


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📄 Resume Job Role Recommender")

st.write(
    "Paste your resume and get the Top 5 job roles "
    "that best match your profile."
)

st.divider()


# ============================================================
# RESUME INPUT
# ============================================================

st.subheader("Enter your resume")

resume_text = st.text_area(
    "Paste your resume text here:",
    height=300,
    placeholder=(
        "Example:\n\n"
        "Python Developer with experience in Python, "
        "Django, SQL, Machine Learning..."
    )
)



# ============================================================
# GET TEXT FROM UPLOADED FILE
# ============================================================


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Recommend Job Roles",
    use_container_width=True
):

    if not resume_text or not resume_text.strip():

        st.warning(
            "Please paste your resume or upload a TXT file."
        )

    else:

        with st.spinner("Analyzing your resume..."):

            results = predict_job_roles(resume_text)

        st.divider()

        st.subheader("🎯 Top 5 Recommended Job Roles")

        for i, (job_role, probability) in enumerate(
            results,
            start=1
        ):

            st.write(
                f"### {i}. {job_role}"
            )

            st.progress(
                min(int(probability), 100)
            )

            st.write(
                f"Match probability: **{probability:.2f}%**"
            )

            st.write("")


# ============================================================
# ABOUT SECTION
# ============================================================

st.divider()

with st.expander("ℹ️ About this project"):

    st.write(
        """
        This application uses Natural Language Processing (NLP)
        and a Bidirectional LSTM model to classify resumes into
        different job-role categories.

        The NLP pipeline includes text preprocessing,
        tokenization, sequence padding and custom Word2Vec
        embeddings.

        The model predicts probabilities for 10 job categories
        and displays the Top 5 recommendations.
        """
    )