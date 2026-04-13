import streamlit as st
from PIL import Image
from groq import Groq
from google.colab import userdata
import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


# -----------------------------
# 1. FOOD CLASSES
# -----------------------------
FOOD_CLASSES = ['apple_pie',
 'baklava',
 'bibimbap',
 'ceviche',
 'cheesecake',
 'chicken_curry',
 'club_sandwich',
 'donuts',
 'dumplings',
 'edamame',
 'escargots',
 'falafel',
 'french_fries',
 'fried_rice',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_dog',
 'hummus',
 'ice_cream',
 'lasagna',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'oysters',
 'pad_thai',
 'pancakes',
 'pho',
 'pizza',
 'sashimi',
 'spaghetti_bolognese',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'tiramisu',
 'waffles']


# -----------------------------
# 2. LOAD KERAS MODEL
# -----------------------------
@st.cache_resource
def load_vision_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {str(e)}")
        return None


# -----------------------------
# 3. LLAMA CULTURAL TRANSLATOR
# -----------------------------
def ask_llama_chef(food_name, user_origin, api_key):
    if not api_key:
        return "❌ Please enter your API Key in the sidebar!"
    
    try:
        client = Groq(api_key=api_key)

        prompt = f"""
        Context: You are a culinary cultural expert. A person from {user_origin} is looking at a dish called "{food_name}".

        Task:
        1. Describe the taste and texture using analogies someone from {user_origin} would understand.
        2. Briefly explain the history of this dish.
        3. Share a fun cultural fact.

        Tone: Friendly and storytelling. English. Under 150 words.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# 4. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Flavor Bridge", page_icon="🌉")
st.title("🌉 Flavor Bridge")
st.markdown("### *Crossing Cultures, One Plate at a Time*")


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    user_home = st.text_input("Where are you from?", "Canada")

    try:
        default_key = userdata.get("GROQ_API_KEY")
    except:
        default_key = ""

    api_key = st.text_input("Groq API Key", value=default_key, type="password")


# -----------------------------
# 5. FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a food photo...", type=["jpg", "jpeg", "png"])


# -----------------------------
# 6. MAIN LOGIC
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Analyze & Translate Culture"):

        model_path = "flavor_bridge_final.keras"
        model = load_vision_model(model_path)

        if model:

            # --- Prediction + Confidence (Keras ResNet50) ---
            preprocess = tf.keras.Sequential([
                tf.keras.layers.Resizing(256, 256),
                tf.keras.layers.CenterCrop(224, 224),
            ])

            img = np.array(image)
            img = preprocess(img)
            img = preprocess_input(img)
            img = tf.expand_dims(img, axis=0)

            pred = model.predict(img)

            pred_idx = np.argmax(pred)
            confidence = float(pred[0][pred_idx]) * 100
            food_name = FOOD_CLASSES[pred_idx].replace("_", " ")

            st.success(f"Detected: **{food_name.title()}** ({confidence:.2f}% confidence)")

            # LLM explanation
            with st.spinner(f"Llama is translating the flavor for someone from {user_home}..."):
                explanation = ask_llama_chef(food_name, user_home, api_key)
                st.info(explanation)

        else:
            st.error("Model file not found or failed to load!")
