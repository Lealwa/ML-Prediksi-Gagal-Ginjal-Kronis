import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

# Set page config should be called at the beginning
st.set_page_config(page_title="Prediksi Gagal Ginjal Kronis", page_icon="🩺", layout="wide")

# Load model
loaded_model = joblib.load('cart_kidney-failure.pkl')

# Fungsi utama aplikasi
def main():
    page = st.sidebar.radio("CART", ["Input", "Visualisasi"])

    if page == "Input":
        # Judul dan deskripsi aplikasi
        st.title("🩺 Prediksi Gagal Ginjal Kronis Menggunakan Metode CART")
        st.markdown("""        
        **Aplikasi ini memprediksi kemungkinan gagal ginjal kronis berdasarkan data medis pasien.**
        
        Silakan masukkan informasi pasien dengan benar dan klik tombol **'Prediksi'** untuk melihat hasil.
        """)
        
        # Divider untuk pemisah bagian
        st.markdown("---")

        # Bagian Input Data Pasien
        st.header("📝 Masukkan Data Pasien")

        # Kolom input untuk tata letak lebih baik
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("🗓️ Umur (Tahun)", min_value=0, value=50, help="Masukkan umur pasien dalam tahun.")
            bp = st.number_input("💓 Tekanan Darah (mmHg)", min_value=0, value=80)
            sg = st.number_input("⚖️ Berat Jenis Urin", min_value=0.0, value=1.015, step=0.001)
            al = st.number_input("💧 Kandungan Albumin dalam Urin", min_value=0, value=0)
            su = st.number_input("🍬 Kandungan Gula dalam Urin", min_value=0, value=0)
            rbc = st.selectbox("🩸 Kondisi Sel Darah Merah", options=["Normal", "Abnormal"])
            rbc_numeric = 1 if rbc == "Normal" else 0

        with col2:
            pc = st.selectbox("🧫 Kondisi Sel Nanah", options=["Normal", "Abnormal"])
            pc_numeric = 1 if pc == "Normal" else 0
            pcc = st.selectbox("🔬 Gumpalan Sel Nanah", options=["Present", "Notpresent"])
            pcc_numeric = 1 if pcc == "Present" else 0
            ba = st.selectbox("🦠 Bakteri dalam Urin", options=["Present", "Notpresent"])
            ba_numeric = 1 if ba == "Present" else 0
            bgr = st.number_input("🩸 Kadar Gula Darah Acak (mg/dL)", min_value=0, value=100)
            bu = st.number_input("🧪 Kadar Urea (mg/dL)", min_value=0, value=30)

        with col3:
            sc = st.number_input("🩸 Kadar Kreatinin (mg/dL)", min_value=0.0, value=1.0, step=0.1)
            sod = st.number_input("🧂 Kadar Natrium (mEq/L)", min_value=0.0, value=135.0, step=0.1)
            pot = st.number_input("🧂 Kadar Kalium (mEq/L)", min_value=0.0, value=4.5, step=0.1)
            hemo = st.number_input("🩸 Kadar Hemoglobin (g/dL)", min_value=0.0, value=12.0, step=0.1)
            pcv = st.number_input("📊 Persentase Sel Darah Merah (%)", min_value=0, value=40)

        # Tambahan Input Data Lainnya
        wbcc = st.number_input("🦠 Jumlah Sel Darah Putih (sel/mm³)", min_value=0, value=8000)
        rbcc = st.number_input("🩸 Jumlah Sel Darah Merah (juta sel/mm³)", min_value=0.0, value=4.5, step=0.1)

        col4, col5 = st.columns(2)

        with col4:
            htn = st.selectbox("💓 Hipertensi", options=["No", "Yes"])
            htn_numeric = 1 if htn == "Yes" else 0
            dm = st.selectbox("🍬 Diabetes", options=["No", "Yes"])
            dm_numeric = 1 if dm == "Yes" else 0
            cad = st.selectbox("❤️ Penyakit Jantung Koroner", options=["No", "Yes"])
            cad_numeric = 1 if cad == "Yes" else 0

        with col5:
            appet = st.selectbox("🍽️ Nafsu Makan", options=["Poor", "Good"])
            appet_numeric = 1 if appet == "Good" else 0
            pe = st.selectbox("🦵 Pembengkakan pada Kaki", options=["No", "Yes"])
            pe_numeric = 1 if pe == "Yes" else 0
            ane = st.selectbox("🩸 Anemia", options=["No", "Yes"])
            ane_numeric = 1 if ane == "Yes" else 0

        # Tombol Prediksi
        if st.button("🔍 Prediksi", use_container_width=True):
            # Prediksi menggunakan model yang dimuat
            prediction = loaded_model.predict([[age, bp, sg, al, su, rbc_numeric, pc_numeric, pcc_numeric,
                                                ba_numeric, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc,
                                                htn_numeric, dm_numeric, cad_numeric, appet_numeric, 
                                                pe_numeric, ane_numeric]])

            # Menampilkan hasil prediksi dengan styling
            if prediction[0] == 'ckd':
                label = "⚠️ Terdiagnosis Gagal Ginjal Kronis"
                color = "red"
            else:
                label = "✅ Tidak Terdiagnosis Gagal Ginjal Kronis"
                color = "green"

            st.markdown(f"<h2 style='color: {color}; text-align: center;'>{label}</h2>", unsafe_allow_html=True)

        # Footer
        st.markdown("""        
        <br><br>
        <hr>
        <div style="text-align: center;">
            <p>© 2024 Aplikasi Prediksi Gagal Ginjal Kronis | Dibuat dengan ❤️ oleh [Nama Anda]</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Visualisasi":
        feature_names = [
            "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", 
            "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", 
            "cad", "appet", "pe", "ane"
        ]
        target_name = ['ckd', 'notckd']

        # Gambar pohon keputusan
        fig = plt.figure(figsize=(20, 20))  # Sesuaikan ukuran gambar
        _ = tree.plot_tree(loaded_model,
                           feature_names=feature_names,
                           class_names=target_name,
                           filled=True)

        # Menampilkan visualisasi
        st.pyplot(fig)

# Jalankan aplikasi
if __name__ == "__main__":
    main()
