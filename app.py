import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("🖊️ Handwritten Digit Recognition App")
st.write("Draw a digit (0-9) and I'll guess it with confidence out of 100!")

# Load pretrained model (you can train and save your own MNIST model)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

# Create drawing canvas
from streamlit_drawable_canvas import st_canvas

st.write("Draw below:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 28, 28, 1)
        img_array = img_array / 255.0

        pred = model.predict(img_array)
        digit = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f} / 100")
    else:
        st.warning("Please draw a digit first.")
