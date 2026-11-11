import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="belajar klasifikasi jeruk", page_icon=":tangerine:")

# Load Otak AI
model = joblib.load('model_klasifikasi_jeruk.joblib')

# Header & Deskripsi
st.title("🍊 belajar klasifikasi jeruk")
st.markdown("aplikasi machine learning classification untuk memprediksi kualitas jeruk")

# Input Numerical (Slider)
diameter = st.slider("diameter", min_value=4.0, max_value=10.0, value=6.5)
berat = st.slider("berat", min_value=100.0, max_value=250.0, value=210.0)
tebal_kulit = st.slider("tebal kulit", min_value=0.2, max_value=1.0, value=0.8)
kadar_gula = st.slider("kadar gula", min_value=8.0, max_value=14.0, value=12.0)

# Input Categorical & Ordinal (Pills)
asal_daerah = st.pills("asal daerah", ["Kalimantan", "Jawa Barat", "Jawa Tengah"], default="Kalimantan")
warna = st.pills("warna", ["hijau", "kuning", "oranye"], default="hijau")
musim_panen = st.pills("musim panen", ["kemarau", "hujan"], default="kemarau")

# Tombol Prediksi
if st.button("prediksi", type="primary"):
    data_baru = pd.DataFrame(
        [[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]],
        columns=['diameter', 'berat', 'tebal_kulit', 'kadar_gula', 'asal_daerah', 'warna', 'musim_panen']
    )
    
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    
    st.success(f"model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase * 100:.2f}%**")
    st.balloons()

st.divider()
st.caption("dibuat dengan ❤️ oleh Fadhil Aufa")