import streamlit as st
import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine

# Konfigurasi Halaman Web
st.set_page_config(page_title="Smart Traffic AI", layout="wide", page_icon="🚦")

st.title("🚦 Smart Traffic Prediction & AI Assistant")
st.write("Aplikasi ini memprediksi kepadatan lalu lintas dan menyediakan AI Assistant untuk analisis.")

# ==========================================
# KONFIGURASI KONEKSI POSTGRESQL
# ==========================================
# Menggunakan password Alamsyah_01
DB_URL = 'postgresql://postgres:Alamsyah_01@localhost:5432/traffic_db'
engine = create_engine(DB_URL)

@st.cache_data(ttl=600) # Cache data selama 10 menit
def load_data_from_db():
    try:
        # Mengambil data langsung dari PostgreSQL
        query = "SELECT * FROM traffic_table"
        df = pd.read_sql(query, engine)
        
        # Preprocessing Waktu
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=cats, ordered=True)
        return df
    except Exception as e:
        st.error(f"Gagal terhubung ke Database: {e}")
        return None

df = load_data_from_db()

# --- MEMBUAT MULTIPAGE / TAB ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Analisis", "🔮 Prediksi Kepadatan", "🤖 Chat AI (Konek Data)"])

# ==========================================
# TAB 1: DASHBOARD ANALISIS (BERBAGAI CHART)
# ==========================================
with tab1:
    if df is not None:
        st.header("Analisis Volume Kendaraan Komprehensif")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data Record", f"{len(df):,}")
        col2.metric("Rata-rata Kendaraan/Jam", f"{int(df['Vehicles'].mean())}")
        col3.metric("Rekor Tertinggi (Maks)", f"{df['Vehicles'].max()}")
        col4.metric("Jumlah Persimpangan", f"{df['Junction'].nunique()}")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("📈 Tren Kepadatan Harian (Time Series)")
            trend_harian = df.groupby('Date')['Vehicles'].mean()
            st.line_chart(trend_harian)
            
        with col_chart2:
            st.subheader("⏰ Rata-rata Kepadatan per Jam (Peak Hours)")
            trend_jam = df.groupby('Hour')['Vehicles'].mean()
            st.area_chart(trend_jam)

        st.markdown("---")

        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            st.subheader("📅 Kepadatan Berdasarkan Hari")
            trend_hari = df.groupby('DayOfWeek')['Vehicles'].mean()
            st.bar_chart(trend_hari)
            
        with col_chart4:
            st.subheader("📍 Perbandingan Antar Junction")
            trend_junction = df.groupby('Junction')['Vehicles'].sum()
            st.bar_chart(trend_junction)

        with st.expander("Lihat Data Mentah dari PostgreSQL (Raw Data)"):
            st.dataframe(df.head(100), use_container_width=True)

# ==========================================
# TAB 2: PREDIKSI TRAFIK
# ==========================================
with tab2:
    st.header("Prediksi Kondisi Lalu Lintas")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameter")
        junction = st.selectbox("Pilih Persimpangan (Junction)", [1, 2, 3, 4])
        hour = st.slider("Jam", 0, 23, 12)
        day_of_week = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
        
        if st.button("Jalankan Prediksi", type="primary"):
            st.session_state.predict = True
    
    with col2:
        st.subheader("Hasil Prediksi")
        # Simulasi prediksi sementara
        base_traffic = 20
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            base_traffic += 60
            
        prediction = int(np.random.normal(loc=base_traffic + (junction*5), scale=10))
        prediction = max(0, prediction)
        
        st.metric(label="Estimasi Jumlah Kendaraan", value=f"{prediction} Kendaraan")
        
        if prediction > 80:
            st.error("⚠️ Kondisi: MACET BERAT")
        elif prediction > 50:
            st.warning("🟡 Kondisi: PADAT MERAYAP")
        else:
            st.success("✅ Kondisi: LANCAR")

# ==========================================
# TAB 3: CHAT AI (OLLAMA)
# ==========================================
with tab3:
    st.header("Asisten AI Analisis Data Trafik")
    
    data_context = ""
    if df is not None:
        data_context = f"""
        Informasi Dataset Lalu Lintas (Diambil dari PostgreSQL):
        - Total baris record data: {len(df)}
        - Rata-rata kendaraan per jam: {int(df['Vehicles'].mean())}
        - Jumlah persimpangan (Junction): {df['Junction'].nunique()}
        - Kendaraan maksimal dalam 1 jam: {df['Vehicles'].max()}
        Tugasmu adalah Data Analyst. Jawab singkat dan padat berdasarkan data di atas.
        """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanya soal data (Contoh: Jam berapa biasanya macet?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_prompt = f"{data_context}\n\nPertanyaan pengguna: {prompt}"
            
            try:
                # Ganti 'llama3' dengan model Ollama yang kamu pakai (misal: 'gemma' atau 'mistral')
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={"model": "qwen3:8b", "prompt": full_prompt, "stream": False},
                    timeout=3000
                )
                
                if response.status_code == 200:
                    ai_reply = response.json()['response']
                else:
                    ai_reply = f"Error dari Ollama: {response.status_code}"
            except requests.exceptions.ConnectionError:
                ai_reply = "Gagal terhubung ke AI. Pastikan Ollama menyala di localhost:11434."
                
            message_placeholder.markdown(ai_reply)
            
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})