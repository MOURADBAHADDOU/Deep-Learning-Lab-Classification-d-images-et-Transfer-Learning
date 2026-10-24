import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input


# =====================================================
# 1. Configuration
# =====================================================

st.set_page_config(
    page_title="CIFAR-10 Model Studio",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATHS = {
    "MLP": str(BASE_DIR / "models/mlp/model_mlp.keras"),
    "CNN A": str(BASE_DIR / "models/cnn_a/cnn_a_cifar10.keras"),
    "CNN B": str(BASE_DIR / "models/cnn_b/cnn_bifar10.keras"),
    "CNN C": str(BASE_DIR / "models/cnn_c/cnn_cifar10.keras"),
    "ResNet50": str(BASE_DIR / "models/resnet50/resnet50_cifar10.keras"),
}

# Résultats de test mesurés précédemment.
# Ils servent de référence, pas de garantie pour une image donnée.
BENCHMARK_ACCURACY = {
    "MLP": 43.36,
    "CNN A": 72.26,
    "CNN B": 77.33,
    "CNN C": 81.91,
    "ResNet50": 90.17,
}


# =====================================================
# 2. Charger les modèles une seule fois
# =====================================================

@st.cache_resource
def load_all_models():
    loaded = {}

    for name, path in MODEL_PATHS.items():
        loaded[name] = keras.models.load_model(path)

    return loaded


# =====================================================
# 3. Prétraiter une image selon le modèle
# =====================================================

def prepare_image(image, model_name):
    if model_name == "ResNet50":
        image = image.resize((224, 224))
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        return preprocess_input(array)

    # MLP et CNN : images 32 x 32, valeurs entre 0 et 1
    image = image.resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return array


# =====================================================
# 4. Faire une prédiction
# =====================================================

def predict_image(model, image, model_name):
    input_array = prepare_image(image, model_name)

    start = time.perf_counter()

    probabilities = model.predict(
        input_array,
        verbose=0
    )[0]

    elapsed = (time.perf_counter() - start) * 1000

    predicted_index = int(np.argmax(probabilities))

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities,
        "time_ms": elapsed,
    }


# =====================================================
# 5. Interface
# =====================================================

st.title("🧠 CIFAR-10 Model Studio")

st.write(
    "Compare five trained deep learning models "
    "for image classification."
)

st.info(
    "The app reuses saved models. No training is performed."
)

# Charger les modèles
try:
    models = load_all_models()
    st.success("All five models loaded successfully.")
except Exception as error:
    st.error(f"Unable to load the models: {error}")
    st.stop()


# Choix du mode
mode = st.radio(
    "Prediction mode",
    [
        "Recommended model",
        "Choose a model",
        "Compare all five models",
    ],
    horizontal=True
)

# Choix de l'image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded image")
        st.image(image, caption="Input image", width=300)

        if st.button("Run prediction", type="primary"):

            if mode == "Recommended model":
                selected_models = ["ResNet50"]

            elif mode == "Choose a model":
                selected = st.selectbox(
                    "Select the model",
                    list(MODEL_PATHS.keys())
                )
                selected_models = [selected]

            else:
                selected_models = list(MODEL_PATHS.keys())

            results = {}

            with st.spinner("Running predictions..."):
                for name in selected_models:
                    results[name] = predict_image(
                        models[name],
                        image,
                        name
                    )

            # Afficher les prédictions
            st.subheader("Prediction results")

            columns = st.columns(len(results))

            for column, (name, result) in zip(
                columns, results.items()
            ):
                with column:
                    st.markdown(f"### {name}")
                    st.metric(
                        "Predicted class",
                        result["class"]
                    )
                    st.metric(
                        "Confidence",
                        f'{result["confidence"]:.2%}'
                    )
                    st.caption(
                        f'Inference time: {result["time_ms"]:.1f} ms'
                    )

            # Probabilités détaillées
            st.subheader("Class probabilities")

            probability_data = {}

            for name, result in results.items():
                probability_data[name] = result["probabilities"]

            import pandas as pd

            probability_df = pd.DataFrame(
                probability_data,
                index=CLASS_NAMES
            )

            st.bar_chart(probability_df)

            # Référence des performances précédentes
            st.subheader("Previously measured test accuracy")

            benchmark_df = pd.DataFrame({
                "Model": list(BENCHMARK_ACCURACY.keys()),
                "Test accuracy (%)": list(
                    BENCHMARK_ACCURACY.values()
                )
            }).set_index("Model")

            st.bar_chart(benchmark_df)

            st.caption(
                "Benchmark accuracy comes from previous evaluations "
                "on CIFAR-10. It does not guarantee that a model "
                "will predict every individual image correctly. "
                "Confidence scores are not necessarily calibrated."
            )

    except Exception as error:
        st.error(f"Error processing the image: {error}")
