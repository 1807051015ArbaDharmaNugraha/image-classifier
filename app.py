import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("model_cnn.h5")

st.title("🧠 Aplikasi Klasifikasi Gambar CNN")
st.write("Unggah gambar dan lihat hasil prediksi model!")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((32,32))
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    prediction = model.predict(img_array)
    label = np.argmax(prediction)

    class_names = ['pesawat', 'mobil', 'burung', 'kucing', 'rusa', 'anjing', 'katak', 'kuda', 'kapal', 'truk']
    st.subheader(f"Prediksi: {class_names[label]}")
