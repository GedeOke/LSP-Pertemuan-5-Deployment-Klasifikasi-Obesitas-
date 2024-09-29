import streamlit as st
import numpy as np
import joblib

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('obs.pkl')
    return model

st.title("Prediksi Status Berat Badan Berdasarkan BMI")
st.write("Aplikasi ini memprediksi status berat badan berdasarkan input umur, jenis kelamin, tinggi badan, dan berat badan menggunakan model yang telah dilatih.")

model = load_model()

st.header("Masukkan Data Anda")

age = st.number_input('Umur (Tahun)', min_value=1, max_value=120, value=25)
gender = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
height = st.number_input('Tinggi Badan (cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Berat Badan (kg)', min_value=1, max_value=300, value=70)

gender_encoded = 1 if gender == 'Laki-laki' else 0

bmi = weight / ((height / 100) ** 2)

input_data = np.array([[age, gender_encoded, height, weight, bmi]])

if st.button('Prediksi'):
    hasil_klasifikasi = model.predict(input_data)
    
    label_dict = {0: 'Berat Badan Normal', 1: 'Obesitas', 2: 'Kelebihan Berat Badan', 3: 'Kurus'}
    
    st.subheader("Hasil Prediksi")
    st.markdown(f"**Status Berat Badan Anda:** {label_dict[hasil_klasifikasi[0]]}")
    
    st.write(f"**BMI Anda:** {bmi:.2f}")