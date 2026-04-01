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
# Step 1: Use the legacy tf_keras package for loading
import tf_keras as keras

# Step 2: Load with compile=False to avoid optimizer-related TypeErrors
model = keras.models.load_model("marine_model.h5", compile=False)

st.title("Marine Plastic Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image")

    # Ensure resizing matches your model's expected input shape
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Use the legacy keras model to predict
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.success("Waste Detected")
    else:
        st.success("Clean Water")

