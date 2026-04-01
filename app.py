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

# 1. Direct Load (Force Keras 2 Legacy)
# 'compile=False' is the most important part to skip the error-causing config
model = keras.models.load_model("marine_model.h5", compile=False)

st.title("Marine Plastic Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image")

    # Preprocessing
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 2. Prediction
    prediction = model.predict(img)
    
    # Get the raw score
    score = prediction[0][0] if prediction.shape[-1] == 1 else np.max(prediction)

    if score > 0.5:
        st.warning(f"Waste Detected (Confidence: {score:.2%})")
    else:
        st.success(f"Clean Water (Confidence: {1-score:.2%})")
