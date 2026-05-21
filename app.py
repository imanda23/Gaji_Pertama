import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Pickled Resources
def load_resources():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

encoders, scaler, model = load_resources()

# 2. Streamlit UI
st.title("Aplikasi Prediksi Gaji Pertama Peserta Vokasi")
st.write("Masukkan data peserta di bawah ini untuk memprediksi gaji pertama.")

# Input fields based on dataset features
col1, col2 = st.columns(2)

with col1:
    jenis_kelamin = st.selectbox("Jenis Kelamin", options=['L', 'P'])
    usia = st.number_input("Usia", min_value=17, max_value=50, value=25)
    pendidikan = st.selectbox("Pendidikan", options=['SMA', 'SMK', 'D3', 'S1'])
    jurusan = st.selectbox("Jurusan", options=['administrasi', 'desain grafis', 'otomotif', 'teknik las', 'teknik listrik'])

with col2:
    durasi_jam = st.number_input("Durasi Pelatihan (Jam)", min_value=1, max_value=100, value=60)
    nilai_ujian = st.number_input("Nilai Ujian", min_value=10.0, max_value=100.0, value=80.0)
    status_bekerja = st.selectbox("Status Bekerja", options=['Belum Bekerja', 'Sudah Bekerja'])

if st.button("Prediksi Gaji"):
    # 3. Preprocessing input
    input_data = pd.DataFrame({
        'Jenis_Kelamin': [jenis_kelamin],
        'Usia': [usia],
        'Pendidikan': [pendidikan],
        'Jurusan': [jurusan],
        'Durasi_Jam': [durasi_jam],
        'Nilai_Ujian': [nilai_ujian],
        'Status_Bekerja': [status_bekerja]
    })

    # Apply Label Encoding
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])

    # Apply Scaling
    # Column order must match training order
    kolom_urutan = ['Jenis_Kelamin', 'Usia', 'Pendidikan', 'Jurusan', 'Durasi_Jam', 'Nilai_Ujian', 'Status_Bekerja']
    input_scaled = scaler.transform(input_data[kolom_urutan])

    # 4. Predict
    prediction = model.predict(input_scaled)

    st.success(f"Hasil Prediksi Gaji Pertama: Rp {prediction[0]:.2f} Juta")
