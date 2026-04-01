import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
#from tensorflow.keras.models import load_model
import tf_keras as keras

#model = load_model("marine_model.h5")
#model = tf.compat.v1.keras.models.load_model("marine_model.h5")
model = keras.models.load_model("marine_model.h5")

st.title("Marine Plastic Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image")

    img = cv2.resize(image, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.success("Waste Detected")
    else:
        st.success("Clean Water")
