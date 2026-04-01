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
# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# import tf_keras as keras
# from tf_keras.models import Sequential
# from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# # 1. MANUALLY DEFINE THE MODEL (Bypasses the 'from_config' crash)
# def build_model():
#     model = Sequential([
#         Input(shape=(224, 224, 3)),
#         Conv2D(32, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'), # Add/Remove these based on YOUR model
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dense(1, activation='sigmoid') 
#     ])
#     return model

# # 2. LOAD WEIGHTS ONLY
# model = build_model()
# try:
#     # 'skip_mismatch=True' allows the app to run even if layers don't match perfectly
#     model.load_weights("marine_model.h5", by_name=True, skip_mismatch=True)
# except Exception as e:
#     st.error("Weight loading failed. Ensure the architecture in build_model() matches your training.")

# st.title("Marine Plastic Detection")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)
#     st.image(image, caption="Uploaded Image")

#     img = cv2.resize(image, (224, 224))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     # 3. PREDICT
#     prediction = model.predict(img)
#     score = float(prediction[0][0])

    # if score > 0.5:
    #     st.warning(f"Waste Detected (Score: {score:.2f})")
    # else:
    #     st.success(f"Clean Water (Score: {score:.2f})")

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tf_keras as keras

# ✅ Load the actual saved model
@st.cache_resource
def load_model():
    model = keras.models.load_model("marine_model.h5")
    model(tf.zeros((1, 224, 224, 3)))  # warm up
    return model

model = load_model()

def gradcam(image_bgr):
    IMG_SIZE = 224

    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # ✅ Build grad_model using base MobileNetV2
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("out_relu")

    grad_model = keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    img_tensor = tf.constant(img_array)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, base_out = grad_model(img_tensor)
        tape.watch(conv_outputs)

        x = model.layers[1](base_out)       # GlobalAveragePooling2D
        predictions = model.layers[2](x)    # Dense

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out = conv_outputs[0].numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_out[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap_color, 0.7, 0)
    return overlay, float(predictions.numpy()[0][0])


# ✅ Streamlit UI
st.title("Marine Plastic Detection 🌊")
st.write("Upload an image to detect marine plastic waste using Grad-CAM")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)  # BGR format

    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original Image")

    with col2:
        with st.spinner("Generating Grad-CAM..."):
            overlay, score = gradcam(image_bgr)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM Output")

    # Prediction result
    if score > 0.5:
        st.warning(f"⚠️ Waste Detected — Confidence: {score*100:.1f}%")
    else:
        st.success(f"✅ Clean Water — Confidence: {(1-score)*100:.1f}%")
# ```

# And update your `requirements.txt`:
# ```
# numpy<2.0.0
# tensorflow==2.15.0
# tf-keras==2.15.0
# opencv-python-headless
# streamlit
# h5py
