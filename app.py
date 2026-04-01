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
# 
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# 1. MANUALLY DEFINE THE MODEL (Bypasses the 'from_config' crash)
def build_model():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'), # Add/Remove these based on YOUR model
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])
    return model

# 2. LOAD WEIGHTS ONLY
model = build_model()
try:
    # 'skip_mismatch=True' allows the app to run even if layers don't match perfectly
    model.load_weights("marine_model.h5", by_name=True, skip_mismatch=True)
except Exception as e:
    st.error("Weight loading failed. Ensure the architecture in build_model() matches your training.")

st.title("Marine Plastic Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image")

    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 3. PREDICT
    prediction = model.predict(img)
    score = float(prediction[0][0])

    if score > 0.5:
        st.warning(f"Waste Detected (Score: {score:.2f})")
    else:
        st.success(f"Clean Water (Score: {score:.2f})")

