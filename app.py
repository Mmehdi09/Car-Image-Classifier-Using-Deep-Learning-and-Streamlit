import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Class Names + Icons
# ---------------------
CLASS_NAMES = ['Audi', 'Hyndai Creta', 'Mahindra Scorpio', 'Rolls Royce',
               'Swift', 'Tata Safari', 'Toyota Innova']

CLASS_ICONS = {
    "Audi": "🚗",
    "Hyndai Creta": "🚙",
    "Mahindra Scorpio": "🚙",
    "Rolls Royce": "🚘",
    "Swift": "🚗",
    "Tata Safari": "🚙",
    "Toyota Innova": "🚐"
}

# ---------------------
# Load Model
# ---------------------
@st.cache_resource
def load_trained_model():
    return load_model("CarModel.h5")

def preprocess_image(image):
    image = image.resize((128,128))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_car_class(model, image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    idx = np.argmax(predictions[0])
    pred_class = CLASS_NAMES[idx]
    confidence = float(predictions[0][idx]) * 100
    return pred_class, confidence, predictions[0], idx

# ---------------------
# UI Config
# ---------------------
st.set_page_config(page_title="Car Image Classifier", page_icon="🚗", layout="wide")
st.title("🚘 Car Image Classifier")
st.markdown("Upload a car image to classify it into categories trained in the model.")
st.markdown("---")

with st.spinner("Loading Model..."):
    model = load_trained_model()
if model is None:
    st.error("Model not loaded. Please check the model file.")
    st.stop()
st.success("Model Loaded!")

# ---------------------
# Upload Image
# ---------------------
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button(" Classify Image", type="primary"):
        with st.spinner("Classifying..."):
            pred_class, confidence, all_preds, pred_idx = predict_car_class(model, image)

        # ---------------------
        # Tabs Layout
        # ---------------------
        tab1, tab2, tab3 = st.tabs(
            ["📌 Classification Results", "📊 Prediction Graph", "🏆 Top 3 Predictions"]
        )

        # --- Tab 1: Results ---
        with tab1:
            st.markdown("""
                <div style="background:#ffffff;border:1px solid #ccc;border-radius:10px;
                padding:15px;margin-bottom:15px;box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h4 style="margin:0;color:#000000;">Classification Results</h4>
                </div>
            """, unsafe_allow_html=True)

            st.success(f"**Predicted Car:** {pred_class}")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

# --- Tab 2: Prediction Graph ---
        with tab2:
            st.markdown("""
                <div style="background:#ffffff;border:1px solid #ccc;border-radius:10px;
                padding:15px;margin-bottom:15px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h4 style="margin:0;color:#000000;">Prediction Probabilities</h4>
                </div>
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots()
            bars = ax.bar(CLASS_NAMES, all_preds * 100, color="skyblue")
            ax.set_ylabel('Probability (%)', color="black")
            ax.set_title('Prediction Probabilities', color="black")
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', color="black")
            ax.tick_params(axis="y", colors="black")

            # Highlight Top 3 with custom colors
            top3_indices = np.argsort(all_preds)[-3:][::-1]
            highlight_colors = ["orange", "green", "purple"]
            for idx, color in zip(top3_indices, highlight_colors):
                bars[idx].set_color(color)

            # Add percentage labels on top of bars
            for bar, prob in zip(bars, all_preds * 100):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + 1,
                        f"{prob:.1f}%", ha='center', va='bottom', color="black", fontsize=9)

            st.pyplot(fig)

            # --- Add Legend Below Graph ---
            st.markdown(
                """
                <div style="display:flex;justify-content:center;margin-top:10px;">
                    <div style="display:flex;align-items:center;margin:0 15px;">
                        <div style="width:15px;height:15px;background:orange;margin-right:5px;"></div>
                        <span style="color:#FFFFFF;">Top 1 Prediction</span>
                    </div>
                    <div style="display:flex;align-items:center;margin:0 15px;">
                        <div style="width:15px;height:15px;background:green;margin-right:5px;"></div>
                        <span style="color:#FFFFFF;">Top 2 Prediction</span>
                    </div>
                    <div style="display:flex;align-items:center;margin:0 15px;">
                        <div style="width:15px;height:15px;background:purple;margin-right:5px;"></div>
                        <span style="color:#FFFFFF;">Top 3 Prediction</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # --- Tab 3: Top 3 Predictions ---
        with tab3:
            st.markdown("""
                <div style="background:#ffffff;border:1px solid #ccc;border-radius:10px;
                padding:15px;margin-bottom:15px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <h4 style="margin:0;color:#000000;">Top 3 Predictions</h4>
                </div>
            """, unsafe_allow_html=True)

            top3_indices = np.argsort(all_preds)[-3:][::-1]
            top3_classes = [CLASS_NAMES[i] for i in top3_indices]
            top3_confidences = [all_preds[i] * 100 for i in top3_indices]

            highlight_colors = ["orange", "green", "purple"]

            for i in range(3):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;justify-content:flex-start;
                    background:#ffffff;border:1px solid #ccc;border-radius:10px;
                    padding:10px;margin:10px 0;box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                        <div style="width:50px;height:50px;border-radius:50%;
                        background:{highlight_colors[i]};display:flex;align-items:center;justify-content:center;
                        margin-right:15px;">
                            <span style="font-size:20px;color:#fff;font-weight:bold;">{i+1}</span>
                        </div>
                        <div style="flex-grow:1;text-align:left;">
                            <h5 style="margin:0;color:#000000;">{top3_classes[i]}</h5>
                            <p style="margin:0;color:#000000;">Confidence: {top3_confidences[i]:.2f}%</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.markdown("Developed by **Muntazir Mehdi** | [GitHub](#) | [LinkedIn](https://www.linkedin.com/in/muntazir-mehdi-3b4b621b3/)")
st.markdown("© 2025 Car Image Classifier App | All rights reserved.")
