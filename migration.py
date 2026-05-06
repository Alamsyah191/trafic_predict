import pandas as pd
from sqlalchemy import create_engine

# Konfigurasi Koneksi (Sesuaikan dengan setting Postgres Anda)
# Format: 'postgresql://username:password@localhost:port/nama_db'
DB_URL = 'postgresql://postgres:Alamsyah_01@localhost:5432/traffic_db'
engine = create_engine(DB_URL)

# Membaca CSV
df = pd.read_csv('dataset/traffic.csv')

# Mengunggah ke tabel bernama 'traffic_table'
df.to_sql('traffic_table', engine, if_exists='replace', index=False)
print("Data berhasil diunggah ke PostgreSQL!")