# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# model = load_model("marine_model.h5")

# st.title("Marine Plastic Detection")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)

#     st.image(image, caption="Uploaded Image")

#     img = cv2.resize(image, (224,224))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     pred = model.predict(img)[0][0]

#     if pred > 0.5:
#         st.success("Waste Detected")
#     else:
#         st.success("Clean Water")
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tf_keras as keras

# 1. Use the legacy loader but skip the optimizer and metadata
@st.cache_resource # This prevents reloading on every click
def load_my_model():
    # 'compile=False' is critical to skip the version-mismatched config
    return keras.models.load_model("marine_model.h5", compile=False)

try:
    model = load_my_model()
    st.sidebar.success("Model loaded!")
except Exception as e:
    st.error("Model loading failed. Please check the logs.")

st.title("Marine Plastic Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (224x224 is standard for these models)
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    # Handle both binary (single value) and categorical (multiple values) outputs
    score = prediction[0][0] if prediction.shape[-1] == 1 else np.max(prediction)

    if score > 0.5:
        st.warning(f"Waste Detected (Confidence: {score:.2%})")
    else:
        st.success(f"Clean Water (Confidence: {1-score:.2%})")
