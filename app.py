# app.py
# Flavor Bridge Streamlit app
# English comments throughout

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "flavor_bridge_final.keras"   # path to your final saved model
CLASS_JSON_PATH = "flavor_bridge_classes.json"  # optional: class list saved at training time
IMG_SHORT_SIDE = 256
IMG_CROP = 224

# ImageNet mean/std used in your training notebook (PyTorch-style normalization)
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# -----------------------------
# Helper: load class list
# -----------------------------
def load_class_list():
    """
    Load class list from JSON if present, otherwise fall back to the
    exact select_food_list used during training and sorted() it.
    """
    if os.path.exists(CLASS_JSON_PATH):
        try:
            with open(CLASS_JSON_PATH, "r") as f:
                classes = json.load(f)
            # ensure it's a list of strings
            if isinstance(classes, list) and all(isinstance(x, str) for x in classes):
                return classes
        except Exception:
            pass

    # Fallback: the original select_food_list used in training (unsorted),
    # then sorted() to match training pipeline.
    select_food_list = [
        'bibimbap', 'gyoza', 'sashimi', 'pad_thai', 'pho',
        'miso_soup', 'edamame', 'spring_rolls', 'sushi', 'dumplings',
        'hummus', 'falafel', 'baklava', 'chicken_curry', 'fried_rice',
        'tacos', 'guacamole', 'ceviche', 'nachos',
        'pizza', 'hamburger', 'hot_dog', 'steak', 'french_fries',
        'grilled_salmon', 'spaghetti_bolognese', 'lasagna', 'club_sandwich',
        'tiramisu', 'cheesecake', 'macarons', 'donuts', 'waffles',
        'pancakes', 'ice_cream', 'apple_pie', 'strawberry_shortcake',
        'mussels', 'oysters', 'escargots'
    ]
    return sorted(select_food_list)

# Load classes once
FOOD_CLASSES = load_class_list()

# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource
def get_model(path):
    """
    Load Keras model and return it. Cached to avoid reloading on every interaction.
    """
    model = load_model(path)
    return model

# -----------------------------
# Preprocessing functions
# -----------------------------
def preprocess_for_inference(pil_image):
    """
    Convert PIL image to tensor and apply the exact eval_transform used in training:
    Resize to 256 -> CenterCrop 224 -> /255 -> (x - mean) / std
    Returns a float32 tensor shape (224,224,3)
    """
    # Convert PIL to numpy array then tensor
    img = tf.convert_to_tensor(np.array(pil_image), dtype=tf.float32)  # (H,W,3)
    # Resize to 256x256
    img = tf.image.resize(img, [IMG_SHORT_SIDE, IMG_SHORT_SIDE])
    # Center crop to 224x224
    offset = (IMG_SHORT_SIDE - IMG_CROP) // 2
    img = img[offset:offset + IMG_CROP, offset:offset + IMG_CROP, :]
    # Normalize exactly like training: /255 then (x - mean)/std
    img = img / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img

# -----------------------------
# LLM helper (Groq)
# -----------------------------
def ask_llama_chef(food_name, user_origin, api_key):
    """
    Call Groq LLM to produce a cultural translation.
    Keep the original behavior; return error message if API key missing.
    """
    if not api_key:
        return "❌ Please enter your API Key in the sidebar!"
    try:
        from groq import Groq
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
        return f"Error calling LLM: {str(e)}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Flavor Bridge", page_icon="🌉")
st.title("🌉 Flavor Bridge")
st.markdown("### *Crossing Cultures, One Plate at a Time*")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configuration")
    user_home = st.text_input("Where are you from?", "Canada")
    api_key = st.text_input("Groq API Key", value="", type="password")
    debug_mode = st.checkbox("Show debug info", value=False)

# File uploader
uploaded_file = st.file_uploader("Upload a food photo...", type=["jpg", "jpeg", "png"])

# Main logic
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze & Translate Culture"):
        # Load model
        try:
            model = get_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Preprocess image exactly like training eval_transform
        try:
            img_tensor = preprocess_for_inference(image)  # (224,224,3)
            img_batch = tf.expand_dims(img_tensor, axis=0)  # (1,224,224,3)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        # Predict
        try:
            pred = model.predict(img_batch, verbose=0)  # shape (1, num_classes)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Debug info
        if debug_mode:
            st.write("Model output shape:", pred.shape)
            st.write("Num classes (FOOD_CLASSES):", len(FOOD_CLASSES))
            st.write("Raw prediction vector (first 10):", pred[0][:10])

        # Sanity check: class count must match model output
        if pred.shape[-1] != len(FOOD_CLASSES):
            st.error(
                f"Class count mismatch: model outputs {pred.shape[-1]} classes but FOOD_CLASSES has {len(FOOD_CLASSES)}."
            )
            st.stop()

        # Top-k results
        topk = 3
        top_idxs = np.argsort(pred[0])[::-1][:topk]
        top_probs = pred[0][top_idxs]

        # Display top-1
        pred_idx = int(top_idxs[0])
        confidence = float(top_probs[0]) * 100.0
        food_name = FOOD_CLASSES[pred_idx].replace("_", " ").title()
        st.success(f"Detected: **{food_name}** ({confidence:.2f}% confidence)")

        # Show Top-3 with simple progress bars
        st.markdown("**Top 3 predictions**")
        for i, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
            label = FOOD_CLASSES[int(idx)].replace("_", " ").title()
            pct = float(prob) * 100.0
            st.write(f"{i+1}. {label} — {pct:.2f}%")
            # progress expects 0.0-1.0
            st.progress(min(max(float(prob), 0.0), 1.0))

        # LLM explanation (optional)
        with st.spinner(f"Llama is translating the flavor for someone from {user_home}..."):
            explanation = ask_llama_chef(food_name, user_home, api_key)
            st.info(explanation)

# Footer: helpful tips
st.markdown("---")
st.markdown(
    "**Tips**: The app uses the same Resize->CenterCrop->Normalize pipeline as training. "
    "If predictions look wrong, enable Debug mode to inspect shapes and raw logits."
)
